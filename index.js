import {
    eventSource,
    event_types,
    extension_prompt_types,
    extension_prompt_roles,
    getCurrentChatId,
    getRequestHeaders,
    is_send_press,
    saveSettingsDebounced,
    setExtensionPrompt,
    substituteParams,
    substituteParamsExtended,
} from '../../../../script.js';
import {
    ModuleWorkerWrapper,
    extension_settings,
    getContext,
    modules,
    renderExtensionTemplateAsync,
    doExtrasFetch,
    getApiUrl,
} from '../../../extensions.js';
import { collapseNewlines, registerDebugFunction } from '../../../power-user.js';
import { getDataBankAttachments, getDataBankAttachmentsForSource, getFileAttachment } from '../../../chats.js';
import { debounce, getStringHash, waitUntilCondition, onlyUnique, splitRecursive, trimToStartSentence, trimToEndSentence } from '../../../utils.js';
import { debounce_timeout } from '../../../constants.js';
import { getSortedEntries } from '../../../world-info.js';
import { textgen_types, textgenerationwebui_settings } from '../../../textgen-settings.js';
import { SlashCommandParser } from '../../../slash-commands/SlashCommandParser.js';
import { SlashCommand } from '../../../slash-commands/SlashCommand.js';
import { ARGUMENT_TYPE, SlashCommandArgument, SlashCommandNamedArgument } from '../../../slash-commands/SlashCommandArgument.js';
import { SlashCommandEnumValue, enumTypes } from '../../../slash-commands/SlashCommandEnumValue.js';
import { slashCommandReturnHelper } from '../../../slash-commands/SlashCommandReturnHelper.js';
import { callGenericPopup, POPUP_RESULT, POPUP_TYPE } from '../../../popup.js';

/**
 * @typedef {object} HashedMessage
 * @property {string} text - The hashed message text
 * @property {number} hash - The hash used as the vector key
 * @property {number} index - The index of the message in the chat
 */

/**
 * @typedef {object} VectorItem
 * @property {string} type - Type of the item ('chat', 'file', 'world_info')
 * @property {string} text - The text content
 * @property {Object} metadata - Additional metadata for the item
 * @property {boolean} selected - Whether the item is selected for vectorization
 */

const MODULE_NAME = 'vectors-enhanced';

export const EXTENSION_PROMPT_TAG = '3_vectors';

const settings = {
    // Master switch - controls all plugin functionality
    master_enabled: true,  // Master switch: controls all functionality of the plugin, disabled by default
    
    // Vector source settings
    source: 'transformers',
    local_model: '',        // Local transformers model name
    vllm_model: '',
    vllm_url: '',
    ollama_model: '',       // Ollama model name
    ollama_url: '',         // Ollama API address
    
    // General vectorization settings
    auto_vectorize: true,
    chunk_size: 1000,
    overlap_percent: 10,
    score_threshold: 0.25,
    force_chunk_delimiter: '',
    
    // Query settings
    enabled: true,      // Enable vector query
    query_messages: 3,  // Number of recent messages to use for query
    max_results: 10,    // Maximum number of results to return
    
    // Injection settings
    template: '<must_know>The following is from a relevant background knowledge base, containing important context, settings, or details:\n{{text}}</must_know>',
    position: extension_prompt_types.IN_PROMPT,
    depth: 2,
    depth_role: extension_prompt_roles.SYSTEM,
    include_wi: false,
    
    // Content tags
    content_tags: {
        chat: 'past_chat',
        file: 'databank',
        world_info: 'world_part'
    },
    
    // Content selection
    selected_content: {
        chat: {
            enabled: false,
            range: { start: 0, end: -1 },
            types: { user: true, assistant: true },
            tags: '', // comma-separated tag names to extract
            include_hidden: false // Whether to include hidden messages
        },
        files: { enabled: false, selected: [] },
        world_info: { enabled: false, selected: {} } // { worldId: [entryIds] }
    },
    
    // Vector tasks management
    vector_tasks: {} // { chatId: [{ taskId, name, timestamp, settings, enabled }] }
};

const moduleWorker = new ModuleWorkerWrapper(synchronizeChat);
const cachedVectors = new Map(); // Cache for vectorized content
let syncBlocked = false;

/**
 * Generates a unique task ID
 * @returns {string} Unique task ID
 */
function generateTaskId() {
    return `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
}

/**
 * Gets all vector tasks for a chat
 * @param {string} chatId Chat ID
 * @returns {Array} Array of tasks
 */
function getChatTasks(chatId) {
    if (!settings.vector_tasks[chatId]) {
        settings.vector_tasks[chatId] = [];
    }
    return settings.vector_tasks[chatId];
}

/**
 * Adds a new vector task
 * @param {string} chatId Chat ID
 * @param {object} task Task object
 */
function addVectorTask(chatId, task) {
    const tasks = getChatTasks(chatId);
    tasks.push(task);
    settings.vector_tasks[chatId] = tasks;
    Object.assign(extension_settings.vectors_enhanced, settings);
    saveSettingsDebounced();
}

/**
 * Removes a vector task
 * @param {string} chatId Chat ID
 * @param {string} taskId Task ID to remove
 */
async function removeVectorTask(chatId, taskId) {
    const tasks = getChatTasks(chatId);
    const index = tasks.findIndex(t => t.taskId === taskId);
    if (index !== -1) {
        // Delete the vector collection
        await purgeVectorIndex(`${chatId}_${taskId}`);
        // Remove from tasks list
        tasks.splice(index, 1);
        settings.vector_tasks[chatId] = tasks;
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
    }
}

/**
 * Gets the Collection ID for a file embedded in the chat.
 * @param {string} fileUrl URL of the file
 * @returns {string} Collection ID
 */
function getFileCollectionId(fileUrl) {
    return `file_${getHashValue(fileUrl)}`;
}

/**
 * Gets the chunk delimiters for splitting text.
 * @returns {string[]} Array of chunk delimiters
 */
function getChunkDelimiters() {
    const delimiters = ['\n\n', '\n', ' ', ''];
    if (settings.force_chunk_delimiter) {
        delimiters.unshift(settings.force_chunk_delimiter);
    }
    return delimiters;
}

/**
 * Splits text into chunks with optional overlap.
 * @param {string} text Text to split
 * @param {number} chunkSize Size of each chunk
 * @param {number} overlapPercent Overlap percentage
 * @returns {string[]} Array of text chunks
 */
function splitTextIntoChunks(text, chunkSize, overlapPercent) {
    const delimiters = getChunkDelimiters();
    const overlapSize = Math.round(chunkSize * overlapPercent / 100);
    const adjustedChunkSize = overlapSize > 0 ? (chunkSize - overlapSize) : chunkSize;
    
    const chunks = splitRecursive(text, adjustedChunkSize, delimiters);
    
    if (overlapSize > 0) {
        return chunks.map((chunk, index) => overlapChunks(chunk, index, chunks, overlapSize));
    }
    
    return chunks;
}

/**
 * Modifies text chunks to include overlap with adjacent chunks.
 * @param {string} chunk Current item
 * @param {number} index Current index
 * @param {string[]} chunks List of chunks
 * @param {number} overlapSize Size of the overlap
 * @returns {string} Overlapped chunks
 */
function overlapChunks(chunk, index, chunks, overlapSize) {
    const halfOverlap = Math.floor(overlapSize / 2);
    const nextChunk = chunks[index + 1];
    const prevChunk = chunks[index - 1];

    const nextOverlap = trimToEndSentence(nextChunk?.substring(0, halfOverlap)) || '';
    const prevOverlap = trimToStartSentence(prevChunk?.substring(prevChunk.length - halfOverlap)) || '';
    const overlappedChunk = [prevOverlap, chunk, nextOverlap].filter(x => x).join(' ');

    return overlappedChunk;
}

/**
 * Extracts content from specific tags in a message
 * @param {string} text Message text
 * @param {string[]} tags Tags to extract
 * @returns {string} Extracted content or original text if no tags specified
 */
function extractTagContent(text, tags) {
    if (!tags || tags.length === 0) return text;
    
    let extractedContent = [];
    
    for (const tag of tags) {
        const regex = new RegExp(`<${tag}>([\s\S]*?)<\/${tag}>`, 'gi');
        const matches = [...text.matchAll(regex)];
        matches.forEach(match => {
            if (match[1]) extractedContent.push(match[1].trim());
        });
    }
    
    return extractedContent.length > 0 ? extractedContent.join('\n\n') : text;
}

/**
 * Gets all vectorizable content based on current settings
 * @returns {Promise<VectorItem[]>} Array of vector items
 */
async function getVectorizableContent() {
    const items = [];
    const context = getContext();
    
    // Chat messages
    if (settings.selected_content.chat.enabled && context.chat) {
        const chatSettings = settings.selected_content.chat;
        const start = chatSettings.range?.start || 0;
        const end = chatSettings.range?.end || -1;
        const types = chatSettings.types || { user: true, assistant: true };
        const tags = chatSettings.tags || '';
        
        const messages = context.chat.slice(start, end === -1 ? undefined : end);
        const tagList = tags ? tags.split(',').map(t => t.trim()).filter(t => t) : [];
        
        messages.forEach((msg, idx) => {
            // Handle hidden messages
            if (msg.is_system === true && !chatSettings.include_hidden) {
                return; // Skip hidden messages unless explicitly included
            }
            
            if (!types.user && msg.is_user) return;
            if (!types.assistant && !msg.is_user) return;
            
            const extractedText = extractTagContent(substituteParams(msg.mes), tagList);
            
            items.push({
                type: 'chat',
                text: extractedText,
                metadata: {
                    index: start + idx,
                    is_user: msg.is_user,
                    name: msg.name,
                    is_hidden: msg.is_system === true
                },
                selected: true
            });
        });
    }
    
    // Files
    if (settings.selected_content.files.enabled) {
        const allFiles = [...getDataBankAttachments(), ...context.chat.filter(x => x.extra?.file).map(x => x.extra.file)];
        
        for (const file of allFiles) {
            if (!settings.selected_content.files.selected.includes(file.url)) continue;
            
            const text = await getFileAttachment(file.url);
            items.push({
                type: 'file',
                text: text,
                metadata: {
                    name: file.name,
                    url: file.url,
                    size: file.size
                },
                selected: true
            });
        }
    }
    
    // World Info
    if (settings.selected_content.world_info.enabled) {
        const entries = await getSortedEntries();
        
        for (const entry of entries) {
            if (!entry.world || !entry.content || entry.disable) continue;
            
            const selectedEntries = settings.selected_content.world_info.selected[entry.world] || [];
            if (!selectedEntries.includes(entry.uid)) continue;
            
            items.push({
                type: 'world_info',
                text: entry.content,
                metadata: {
                    world: entry.world,
                    uid: entry.uid,
                    key: entry.key.join(', '),
                    comment: entry.comment
                },
                selected: true
            });
        }
    }
    
    return items;
}

/**
 * Updates progress display
 * @param {number} current Current progress
 * @param {number} total Total items
 * @param {string} message Progress message
 */
function updateProgress(current, total, message) {
    const percent = Math.round((current / total) * 100);
    $('#vectors_enhanced_progress').show();
    $('#vectors_enhanced_progress .progress-bar-inner').css('width', `${percent}%`);
    $('#vectors_enhanced_progress .progress-text').text(`${message} (${current}/${total})`);
}

/**
 * Hides progress display
 */
function hideProgress() {
    $('#vectors_enhanced_progress').hide();
    $('#vectors_enhanced_progress .progress-bar-inner').css('width', '0%');
    $('#vectors_enhanced_progress .progress-text').text('Preparing...');
}

/**
 * Generates a task name based on settings
 * @param {object} chatSettings Chat settings
 * @param {number} itemCount Number of items
 * @returns {Promise<string>} Task name
 */
async function generateTaskName(chatSettings, itemCount) {
    const parts = [];
    
    // Chat range
    if (settings.selected_content.chat.enabled) {
        const start = chatSettings.range?.start || 0;
        const end = chatSettings.range?.end || -1;
        if (end === -1) {
            parts.push(`Message #${start} to end`);
        } else {
            parts.push(`Message #${start}-${end}`);
        }
    }
    
    // Files - only count if enabled
    if (settings.selected_content.files.enabled) {
        const fileCount = settings.selected_content.files.selected.length;
        if (fileCount > 0) {
            parts.push(`${fileCount} files`);
        }
    }
    
    // World info - only count if enabled
    if (settings.selected_content.world_info.enabled) {
        const wiCount = Object.values(settings.selected_content.world_info.selected).flat().length;
        if (wiCount > 0) {
            parts.push(`${wiCount} world info entries`);
        }
    }
    
    // If no specific content selected, use generic name
    if (parts.length === 0) {
        parts.push(`${itemCount} items`);
    }
    
    // Add timestamp
    const time = new Date().toLocaleTimeString('en-US', { hour: '2-digit', minute: '2-digit' });
    return `${parts.join(', ')} (${time})`;
}

/**
 * Updates the task list UI
 */
async function updateTaskList() {
    const chatId = getCurrentChatId();
    if (!chatId) return;
    
    const tasks = getChatTasks(chatId);
    const taskList = $('#vectors_enhanced_task_list');
    taskList.empty();
    
    if (tasks.length === 0) {
        taskList.append('<div class="text-muted">No vectorization tasks</div>');
        return;
    }
    
    tasks.forEach((task, index) => {
        const taskDiv = $('<div class="vector-enhanced-task-item"></div>');
        
        const checkbox = $(`
            <label class="checkbox_label flex-container alignItemsCenter">
                <input type="checkbox" ${task.enabled ? 'checked' : ''} />
                <span class="flex1">
                    <strong>${task.name}</strong>
                    <small class="text-muted"> - ${new Date(task.timestamp).toLocaleString('en-US')}</small>
                </span>
            </label>
        `);
        
        checkbox.find('input').on('change', function() {
            task.enabled = this.checked;
            Object.assign(extension_settings.vectors_enhanced, settings);
            saveSettingsDebounced();
        });
        
        const deleteBtn = $(`<button class="menu_button menu_button_icon" title="Delete this task">
            <i class="fa-solid fa-trash"></i>
        </button>`);
        
        deleteBtn.on('click', async () => {
            const confirm = await callGenericPopup('Are you sure you want to delete this vectorization task?', POPUP_TYPE.CONFIRM);
            if (confirm === POPUP_RESULT.AFFIRMATIVE) {
                await removeVectorTask(chatId, task.taskId);
                await updateTaskList();
                toastr.success('Task deleted');
            }
        });
        
        taskDiv.append(checkbox);
        taskDiv.append(deleteBtn);
        taskList.append(taskDiv);
    });
}

/**
 * Vectorizes selected content
 * @returns {Promise<void>}
 */
async function vectorizeContent() {
    const items = await getVectorizableContent();
    if (items.length === 0) {
        toastr.warning('No content selected for vectorization');
        return;
    }
    
    const chatId = getCurrentChatId();
    if (!chatId) {
        toastr.error('No chat selected');
        return;
    }
    
    // Generate task name
    const context = getContext();
    const chatSettings = settings.selected_content.chat;
    const taskName = await generateTaskName(chatSettings, items.length);
    
    try {
        toastr.info('Vectorization started...', 'Processing');
        updateProgress(0, items.length, 'Preparing for vectorization');
        
        // Create new task
        const taskId = generateTaskId();
        const task = {
            taskId: taskId,
            name: taskName,
            timestamp: Date.now(),
            settings: JSON.parse(JSON.stringify(settings.selected_content)),
            enabled: true,
            itemCount: items.length
        };
        
        // Use task-specific collection ID
        const collectionId = `${chatId}_${taskId}`;
        
        // Process items in chunks
        const allChunks = [];
        let processedItems = 0;
        
        for (const item of items) {
            const chunks = splitTextIntoChunks(item.text, settings.chunk_size, settings.overlap_percent);
            chunks.forEach((chunk, idx) => {
                allChunks.push({
                    hash: getHashValue(chunk),
                    text: chunk,
                    index: allChunks.length,
                    metadata: {
                        ...item.metadata,
                        type: item.type,
                        chunk_index: idx,
                        chunk_total: chunks.length
                    }
                });
            });
            
            processedItems++;
            updateProgress(processedItems, items.length, 'Processing content');
        }
        
        // Insert vectors in batches
        updateProgress(0, allChunks.length, 'Inserting vectors');
        const batchSize = 50;
        for (let i = 0; i < allChunks.length; i += batchSize) {
            const batch = allChunks.slice(i, Math.min(i + batchSize, allChunks.length));
            await insertVectorItems(collectionId, batch);
            updateProgress(Math.min(i + batchSize, allChunks.length), allChunks.length, 'Inserting vectors');
        }
        
        // Add task to list
        addVectorTask(chatId, task);
        
        // Update cache for this task
        cachedVectors.set(collectionId, {
            timestamp: Date.now(),
            items: allChunks,
            settings: JSON.parse(JSON.stringify(settings))
        });
        
        hideProgress();
        toastr.success(`Successfully created vectorization task "${taskName}": ${items.length} items, ${allChunks.length} chunks`, 'Success');
        
        // Refresh task list UI
        await updateTaskList();
    } catch (error) {
        console.error('Vectorization failed:', error);
        hideProgress();
        toastr.error('Failed to vectorize content', 'Error');
    }
}

/**
 * Exports vectorized content
 * @returns {Promise<void>}
 */
async function exportVectors() {
    const context = getContext();
    const chatId = getCurrentChatId();
    
    if (!chatId) {
        toastr.error('No chat selected');
        return;
    }
    
    const items = await getVectorizableContent();
    if (items.length === 0) {
        toastr.warning('No content selected for export');
        return;
    }
    
    // Build export content
    let exportText = `Character Card: ${context.name || 'Unknown'}\n`;
    exportText += `Time: ${new Date().toLocaleString('en-US')}\n\n`;
    
    // Group items by type
    const grouped = items.reduce((acc, item) => {
        if (!acc[item.type]) acc[item.type] = [];
        acc[item.type].push(item);
        return acc;
    }, {});
    
    // Files
    exportText += '=== Databank Files ===\n';
    if (grouped.file && grouped.file.length > 0) {
        grouped.file.forEach(item => {
            exportText += `File Name: ${item.metadata.name}\n`;
            exportText += `Content:\n${item.text}\n\n`;
        });
    } else {
        exportText += 'None\n\n';
    }
    
    // World Info
    exportText += '=== World Info ===\n';
    if (grouped.world_info && grouped.world_info.length > 0) {
        grouped.world_info.forEach(item => {
            exportText += `World: ${item.metadata.world}\n`;
            exportText += `Comment: ${item.metadata.comment || 'None'}\n`;
            exportText += `Content: ${item.text}\n\n`;
        });
    } else {
        exportText += 'None\n\n';
    }
    
    // Chat messages
    exportText += '=== Chat History ===\n';
    if (grouped.chat && grouped.chat.length > 0) {
        grouped.chat.forEach(item => {
            exportText += `#${item.metadata.index}: ${item.text}\n\n`;
        });
    } else {
        exportText += 'None\n\n';
    }
    
    // Create and download file
    const blob = new Blob([exportText], { type: 'text/plain;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `VectorExport_${context.name || chatId}_${Date.now()}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    
    toastr.success('Export successful');
}

/**
 * Previews vectorizable content
 * @returns {Promise<void>}
 */
async function previewContent() {
    const items = await getVectorizableContent();
    if (items.length === 0) {
        toastr.warning('No content selected for preview');
        return;
    }
    
    // Group by type
    const grouped = items.reduce((acc, item) => {
        if (!acc[item.type]) acc[item.type] = [];
        acc[item.type].push(item);
        return acc;
    }, {});
    
    let html = '<div class="vector-preview" style="max-height: none; overflow: visible;">';
    html += `<div class="preview-header" style="font-size: 1.2em; font-weight: bold; margin-bottom: 1rem; color: var(--SmartThemeQuoteColor);">Selected Content (${items.length} items)</div>`;
    html += '<div class="preview-sections" style="display: flex; gap: 1rem; overflow: visible;">';
    
    // Files section
    html += '<div class="preview-section" style="flex: 1; display: flex; flex-direction: column; border: 1px solid var(--SmartThemeBorderColor); border-radius: 6px; overflow: hidden; max-height: 60vh;">';
    html += `<div class="preview-section-title" style="padding: 0.75rem; background-color: var(--SmartThemeQuoteColor); color: var(--SmartThemeBodyColor); border-bottom: 1px solid var(--SmartThemeBorderColor); font-weight: bold;">Files (${grouped.file?.length || 0})</div>`;
    html += '<div class="preview-section-content" style="flex: 1; overflow-y: auto; padding: 1rem;">';
    if (grouped.file && grouped.file.length > 0) {
        html += '<div style="text-align: left;">';
        grouped.file.forEach((item, index) => {
            const sizeKB = (item.metadata.size / 1024).toFixed(1);
            html += `<div style="margin-bottom: 0.5rem;">`;
            html += `<strong>${item.metadata.name}</strong> - ${sizeKB} KB`;
            html += `</div>`;
            if (index < grouped.file.length - 1) {
                html += '<hr style="margin: 0.5rem 0; border-color: var(--SmartThemeBorderColor);">';
            }
        });
        html += '</div>';
    } else {
        html += '<div class="preview-empty" style="color: var(--SmartThemeEmColor); font-style: italic;">No files</div>';
    }
    html += '</div></div>';
    
    // World Info section
    html += '<div class="preview-section" style="flex: 1; display: flex; flex-direction: column; border: 1px solid var(--SmartThemeBorderColor); border-radius: 6px; overflow: hidden; max-height: 60vh;">';
    html += `<div class="preview-section-title" style="padding: 0.75rem; background-color: var(--SmartThemeQuoteColor); color: var(--SmartThemeBodyColor); border-bottom: 1px solid var(--SmartThemeBorderColor); font-weight: bold;">World Info (${grouped.world_info?.length || 0})</div>`;
    html += '<div class="preview-section-content" style="flex: 1; overflow-y: auto; padding: 1rem;">';
    if (grouped.world_info && grouped.world_info.length > 0) {
        html += '<div style="text-align: left;">';
        
        // Group world info items by world book
        const worldGroups = {};
        grouped.world_info.forEach(item => {
            const worldName = item.metadata.world;
            if (!worldGroups[worldName]) {
                worldGroups[worldName] = [];
            }
            worldGroups[worldName].push(item);
        });
        
        let worldIndex = 0;
        Object.keys(worldGroups).forEach(worldName => {
            html += `<div style="margin-bottom: 1rem;">`;
            html += `<strong style="color: var(--SmartThemeQuoteColor); font-size: 1.1em;">${worldName}</strong><br>`;
            
            worldGroups[worldName].forEach((item, index) => {
                html += `<div style="margin-left: 1rem; margin-top: 0.25rem; margin-bottom: 0.5rem; padding-left: 0.5rem; border-left: 2px solid var(--SmartThemeBorderColor);">`;
                html += `<span style="display: block; line-height: 1.4;">${item.metadata.comment || '(No comment)'}</span>`;
                html += `</div>`;
            });
            
            html += `</div>`;
            if (worldIndex < Object.keys(worldGroups).length - 1) {
                html += '<hr style="margin: 0.5rem 0; border-color: var(--SmartThemeBorderColor);">';
            }
            worldIndex++;
        });
        
        html += '</div>';
    } else {
        html += '<div class="preview-empty" style="color: var(--SmartThemeEmColor); font-style: italic;">No world info</div>';
    }
    html += '</div></div>';
    
    // Chat messages section
    html += '<div class="preview-section" style="flex: 1; display: flex; flex-direction: column; border: 1px solid var(--SmartThemeBorderColor); border-radius: 6px; overflow: hidden; max-height: 60vh;">';
    html += `<div class="preview-section-title" style="padding: 0.75rem; background-color: var(--SmartThemeQuoteColor); color: var(--SmartThemeBodyColor); border-bottom: 1px solid var(--SmartThemeBorderColor); font-weight: bold;">Chat History (${grouped.chat?.length || 0} messages)</div>`;
    html += '<div class="preview-section-content" style="flex: 1; overflow-y: auto; padding: 1rem;">';
    if (grouped.chat && grouped.chat.length > 0) {
        html += '<div style="text-align: left;">';
        grouped.chat.forEach((item, index) => {
            const msgType = item.metadata.is_user ? 'User' : 'AI';
            html += `<div style="margin-bottom: 0.5rem;">`;
            html += `<strong style="color: var(--SmartThemeQuoteColor);">#${item.metadata.index} - ${msgType} (${item.metadata.name})${item.metadata.is_hidden ? ' [Hidden]' : ''}</strong><br>`;
            html += `<div style="margin-top: 0.5rem; white-space: pre-wrap; word-break: break-word;">${item.text}</div>`;
            html += `</div>`;
            if (index < grouped.chat.length - 1) {
                html += '<hr style="margin: 0.5rem 0; border-color: var(--SmartThemeBorderColor);">';
            }
        });
        html += '</div>';
    } else {
        html += '<div class="preview-empty" style="color: var(--SmartThemeEmColor); font-style: italic;">No chat history</div>';
    }
    html += '</div></div>';
    
    html += '</div></div>';
    
    await callGenericPopup(html, POPUP_TYPE.TEXT, '', {
        okButton: 'Close',
        wide: true,
        large: true
    });
}

/**
 * Cache object for storing hash values
 * @type {Map<string, number>}
 */
const hashCache = new Map();

/**
 * Gets the hash value for a given string
 * @param {string} str Input string
 * @returns {number} Hash value
 */
function getHashValue(str) {
    if (hashCache.has(str)) {
        return hashCache.get(str);
    }
    const hash = getStringHash(str);
    hashCache.set(str, hash);
    return hash;
}

/**
 * Synchronizes chat vectors
 * @param {number} batchSize Batch size for processing
 * @returns {Promise<number>} Number of remaining items
 */
async function synchronizeChat(batchSize = 5) {
    // Check if master switch is enabled
    if (!settings.master_enabled) {
        return -1;
    }

    if (!settings.auto_vectorize) {
        return -1;
    }

    try {
        await waitUntilCondition(() => !syncBlocked && !is_send_press, 1000);
    } catch {
        console.log('Vectors: Synchronization blocked by another process');
        return -1;
    }

    try {
        syncBlocked = true;
        // Auto-vectorization logic will be implemented based on settings
        return -1;
    } finally {
        syncBlocked = false;
    }
}

/**
 * Retrieves vectorized content for injection
 * @param {object[]} chat Chat messages
 * @param {number} contextSize Context size
 * @param {function} abort Abort function
 * @param {string} type Generation type
 */
async function rearrangeChat(chat, contextSize, abort, type) {
    try {
        if (type === 'quiet') {
            console.debug('Vectors: Skipping quiet prompt');
            return;
        }

        setExtensionPrompt(EXTENSION_PROMPT_TAG, '', settings.position, settings.depth, settings.include_wi, settings.depth_role);

        // Check if master switch is enabled
        if (!settings.master_enabled) {
            console.debug('Vectors: Master switch disabled, skipping all functionality');
            return;
        }

        // Check if vector query is enabled
        if (!settings.enabled) {
            console.debug('Vectors: Query disabled by user');
            return;
        }

        const chatId = getCurrentChatId();
        if (!chatId) {
            console.debug('Vectors: No chat ID available');
            return;
        }

        // Query vectors based on recent messages
        const queryMessages = Math.min(settings.query_messages || 3, chat.length);
        const queryText = chat.slice(-queryMessages).map(x => x.mes).join('\n');
        if (!queryText.trim()) return;

        // Get all enabled tasks for this chat
        const tasks = getChatTasks(chatId).filter(t => t.enabled);
        if (tasks.length === 0) {
            console.debug('Vectors: No enabled tasks for this chat');
            return;
        }

        // Query all enabled tasks
        const allResults = [];
        for (const task of tasks) {
            const collectionId = `${chatId}_${task.taskId}`;
            try {
                const results = await queryCollection(collectionId, queryText, settings.max_results || 10);
                console.debug(`Vectors: Query results for task ${task.name}:`, results);
                
                // Process results based on API response structure
                if (results) {
                    // If API returns items array (with text)
                    if (results.items && Array.isArray(results.items)) {
                        results.items.forEach(item => {
                            if (item.text) {
                                allResults.push({
                                    text: item.text,
                                    score: item.score || 0,
                                    metadata: {
                                        ...item.metadata,
                                        taskName: task.name,
                                        taskId: task.taskId
                                    }
                                });
                            }
                        });
                    }
                    // If API only returns hashes and metadata, try to get from cache
                    else if (results.hashes && results.metadata) {
                        const cachedData = cachedVectors.get(collectionId);
                        if (cachedData && cachedData.items) {
                            results.hashes.forEach((hash, index) => {
                                const cachedItem = cachedData.items.find(item => item.hash === hash);
                                if (cachedItem && cachedItem.text) {
                                    allResults.push({
                                        text: cachedItem.text,
                                        score: results.metadata[index]?.score || 0,
                                        metadata: {
                                            ...cachedItem.metadata,
                                            ...(results.metadata[index] || {}),
                                            taskName: task.name,
                                            taskId: task.taskId
                                        }
                                    });
                                }
                            });
                        } else {
                            console.warn(`Vectors: No cached data for collection ${collectionId}, cannot retrieve text`);
                        }
                    }
                }
            } catch (error) {
                console.error(`Vectors: Failed to query task ${task.name}:`, error);
            }
        }

        if (allResults.length === 0) {
            console.debug('Vectors: No query results found');
            return;
        }

        console.debug(`Vectors: Found ${allResults.length} total results`);

        // Sort by score and take top results
        allResults.sort((a, b) => (b.score || 0) - (a.score || 0));
        const topResults = allResults.slice(0, settings.max_results || 10);

        console.debug(`Vectors: Using top ${topResults.length} results`);

        // Group results by type
        const groupedResults = {};
        topResults.forEach(result => {
            const type = result.metadata?.type || 'unknown';
            if (!groupedResults[type]) {
                groupedResults[type] = [];
            }
            groupedResults[type].push(result);
        });

        console.debug('Vectors: Grouped results by type:', Object.keys(groupedResults).map(k => `${k}: ${groupedResults[k].length}`));

        // Format results with tags
        const formattedParts = [];
        
        // Process chat messages
        if (groupedResults.chat && groupedResults.chat.length > 0) {
            const chatTexts = groupedResults.chat
                .sort((a, b) => (a.metadata?.index || 0) - (b.metadata?.index || 0))
                .map(m => m.text)
                .filter(onlyUnique)
                .join('\n\n');
            
            const tag = settings.content_tags?.chat || 'past_chat';
            formattedParts.push(`<${tag}>\n${chatTexts}\n<\/${tag}>`);
        }
        
        // Process world info
        if (groupedResults.world_info && groupedResults.world_info.length > 0) {
            const wiTexts = groupedResults.world_info
                .map(m => m.text)
                .filter(onlyUnique)
                .join('\n\n');
            
            const tag = settings.content_tags?.world_info || 'world_part';
            formattedParts.push(`<${tag}>\n${wiTexts}\n<\/${tag}>`);
        }
        
        // Process files
        if (groupedResults.file && groupedResults.file.length > 0) {
            const fileTexts = groupedResults.file
                .map(m => m.text)
                .filter(onlyUnique)
                .join('\n\n');
            
            const tag = settings.content_tags?.file || 'databank';
            formattedParts.push(`<${tag}>\n${fileTexts}\n<\/${tag}>`);
        }

        // Join all parts
        const relevantTexts = formattedParts.join('\n\n');

        console.debug(`Vectors: Formatted ${formattedParts.length} parts, total length: ${relevantTexts.length}`);

        if (!relevantTexts) {
            console.debug('Vectors: No relevant texts found after formatting');
            return;
        }

        const insertedText = substituteParamsExtended(settings.template, { text: relevantTexts });
        console.debug(`Vectors: Final injected text length: ${insertedText.length}`);
        
        setExtensionPrompt(EXTENSION_PROMPT_TAG, insertedText, settings.position, settings.depth, settings.include_wi, settings.depth_role);
    } catch (error) {
        console.error('Vectors: Failed to rearrange chat', error);
    }
}

window['vectors_rearrangeChat'] = rearrangeChat;

// Global event bindings - ensure buttons are always effective
$(document).on('click', '#vectors_enhanced_preview', async function(e) {
    e.preventDefault();
    console.log('Preview button clicked (global binding)');
    
    if (!settings.master_enabled) {
        toastr.warning('Please enable Chat History Super Manager first');
        return;
    }
    
    try {
        await previewContent();
    } catch (error) {
        console.error('Preview error:', error);
        toastr.error('Preview failed: ' + error.message);
    }
});

$(document).on('click', '#vectors_enhanced_export', async function(e) {
    e.preventDefault();
    console.log('Export button clicked (global binding)');
    
    if (!settings.master_enabled) {
        toastr.warning('Please enable Chat History Super Manager first');
        return;
    }
    
    try {
        await exportVectors();
    } catch (error) {
        console.error('Export error:', error);
        toastr.error('Export failed: ' + error.message);
    }
});

$(document).on('click', '#vectors_enhanced_vectorize', async function(e) {
    e.preventDefault();
    console.log('Vectorize button clicked (global binding)');
    
    if (!settings.master_enabled) {
        toastr.warning('Please enable Chat History Super Manager first');
        return;
    }
    
    try {
        await vectorizeContent();
    } catch (error) {
        console.error('Vectorization error:', error);
        toastr.error('Vectorization failed: ' + error.message);
    }
});

/**
 * Gets request body for vector operations
 * @param {object} args Additional arguments
 * @returns {object} Request body
 */
function getVectorsRequestBody(args = {}) {
    const body = Object.assign({}, args);
    
    switch (settings.source) {
        case 'transformers':
            // Local transformers - read from SillyTavern API settings
            // Try to get model from various possible settings locations
            if (textgenerationwebui_settings.model || textgenerationwebui_settings.embeddings_model) {
                body.model = textgenerationwebui_settings.embeddings_model || textgenerationwebui_settings.model;
            }
            // Also try to get from local transformers settings if available
            if (settings.local_model) {
                body.model = settings.local_model;
            }
            break;
        case 'vllm':
            body.apiUrl = settings.vllm_url || textgenerationwebui_settings.server_urls[textgen_types.VLLM];
            body.model = settings.vllm_model;
            break;
        case 'ollama':
            body.apiUrl = settings.ollama_url || 'http://localhost:11434';
            body.model = settings.ollama_model;
            break;
    }
    
    body.source = settings.source;
    console.debug(`Vectors: Request body for ${settings.source}:`, body);
    return body;
}

/**
 * Throws if the vector source is invalid
 */
function throwIfSourceInvalid() {
    if (settings.source === 'vllm') {
        if (!settings.vllm_url && !textgenerationwebui_settings.server_urls[textgen_types.VLLM]) {
            throw new Error('vLLM URL not configured');
        }
        if (!settings.vllm_model) {
            throw new Error('vLLM model not specified');
        }
    } else if (settings.source === 'ollama') {
        if (!settings.ollama_model) {
            throw new Error('Ollama model not specified');
        }
        // ollama_url is optional because it has a default value of http://localhost:11434
    }
}

/**
 * Gets saved hashes for a collection
 * @param {string} collectionId Collection ID
 * @returns {Promise<number[]>} Array of hashes
 */
async function getSavedHashes(collectionId) {
    const response = await fetch('/api/vector/list', {
        method: 'POST',
        headers: getRequestHeaders(),
        body: JSON.stringify({
            ...getVectorsRequestBody(),
            collectionId: collectionId,
        }),
    });

    if (!response.ok) {
        throw new Error(`Failed to get saved hashes for collection ${collectionId}`);
    }

    return await response.json();
}

/**
 * Inserts vector items into a collection
 * @param {string} collectionId Collection ID
 * @param {object[]} items Items to insert
 * @returns {Promise<void>}
 */
async function insertVectorItems(collectionId, items) {
    throwIfSourceInvalid();

    const response = await fetch('/api/vector/insert', {
        method: 'POST',
        headers: getRequestHeaders(),
        body: JSON.stringify({
            ...getVectorsRequestBody(),
            collectionId: collectionId,
            items: items,
        }),
    });

    if (!response.ok) {
        throw new Error(`Failed to insert vector items for collection ${collectionId}`);
    }
}

/**
 * Queries a collection
 * @param {string} collectionId Collection ID
 * @param {string} searchText Search text
 * @param {number} topK Number of results
 * @returns {Promise<{hashes: number[], metadata: object[]}>}
 */
async function queryCollection(collectionId, searchText, topK) {
    const response = await fetch('/api/vector/query', {
        method: 'POST',
        headers: getRequestHeaders(),
        body: JSON.stringify({
            ...getVectorsRequestBody(),
            collectionId: collectionId,
            searchText: searchText,
            topK: topK,
            threshold: settings.score_threshold,
            includeText: true, // Request to include text content
        }),
    });

    if (!response.ok) {
        throw new Error(`Failed to query collection ${collectionId}`);
    }

    const result = await response.json();
    console.debug(`Vectors: Raw query result for ${collectionId}:`, result);
    return result;
}

/**
 * Purges a vector index
 * @param {string} collectionId Collection ID
 * @returns {Promise<boolean>} Success status
 */
async function purgeVectorIndex(collectionId) {
    try {
        const response = await fetch('/api/vector/purge', {
            method: 'POST',
            headers: getRequestHeaders(),
            body: JSON.stringify({
                ...getVectorsRequestBody(),
                collectionId: collectionId,
            }),
        });

        if (!response.ok) {
            throw new Error(`Could not delete vector index for collection ${collectionId}`);
        }

        console.log(`Vectors: Purged vector index for collection ${collectionId}`);
        cachedVectors.delete(collectionId);
        return true;
    } catch (error) {
        console.error('Vectors: Failed to purge', error);
        return false;
    }
}

/**
 * Updates UI based on settings
 */
function toggleSettings() {
    $('#vectors_enhanced_vllm_settings').toggle(settings.source === 'vllm');
    $('#vectors_enhanced_local_settings').toggle(settings.source === 'transformers');
    $('#vectors_enhanced_ollama_settings').toggle(settings.source === 'ollama');
}

/**
 * Updates UI state based on master switch
 */
function updateMasterSwitchState() {
    const isEnabled = settings.master_enabled;
    
    // Control visibility of main settings areas
    $('#vectors_enhanced_main_settings').toggle(isEnabled);
    $('#vectors_enhanced_content_settings').toggle(isEnabled);
    $('#vectors_enhanced_tasks_settings').toggle(isEnabled);
    $('#vectors_enhanced_actions_settings').toggle(isEnabled);
    
    // If disabled, also disable all input controls (as extra protection), but keep the master switch available
    const settingsContainer = $('#vectors_enhanced_container');
    if (isEnabled) {
        // When enabled, enable all controls
        settingsContainer.find('input, select, textarea, button').prop('disabled', false);
    } else {
        // When disabled, disable all controls except the master switch
        settingsContainer.find('input, select, textarea, button').not('#vectors_enhanced_master_enabled').prop('disabled', true);
        // Ensure the master switch is always enabled
        $('#vectors_enhanced_master_enabled').prop('disabled', false);
    }
    
    // Update visual effect
    if (isEnabled) {
        settingsContainer.removeClass('vectors-disabled');
    } else {
        settingsContainer.addClass('vectors-disabled');
    }
}

/**
 * Updates content selection UI
 */
function updateContentSelection() {
    // This will be called when settings change to update the UI
    $('#vectors_enhanced_chat_settings').toggle(settings.selected_content.chat.enabled);
    $('#vectors_enhanced_files_settings').toggle(settings.selected_content.files.enabled);
    $('#vectors_enhanced_wi_settings').toggle(settings.selected_content.world_info.enabled);
}

/**
 * Updates the file list UI
 */
async function updateFileList() {
    const context = getContext();
    const allFiles = [
        ...getDataBankAttachments(),
        ...(context.chat?.filter(x => x.extra?.file).map(x => x.extra.file) || [])
    ];
    
    const fileList = $('#vectors_enhanced_files_list');
    fileList.empty();
    
    if (allFiles.length === 0) {
        fileList.append('<div class="text-muted">No available files</div>');
        return;
    }
    
    // Group files by source
    const dataBankFiles = getDataBankAttachments();
    const chatFiles = context.chat?.filter(x => x.extra?.file).map(x => x.extra.file) || [];
    
    if (dataBankFiles.length > 0) {
        fileList.append('<div class="file-group-header">Databank Files</div>');
        dataBankFiles.forEach(file => {
            const isChecked = settings.selected_content.files.selected.includes(file.url);
            const checkbox = $(`
                <label class="checkbox_label flex-container alignItemsCenter" title="${file.name}">
                    <input type="checkbox" value="${file.url}" ${isChecked ? 'checked' : ''} />
                    <span class="flex1 text-overflow-ellipsis">${file.name} (${(file.size / 1024).toFixed(1)} KB)</span>
                </label>
            `);
            
            checkbox.find('input').on('change', function() {
                if (this.checked) {
                    if (!settings.selected_content.files.selected.includes(file.url)) {
                        settings.selected_content.files.selected.push(file.url);
                    }
                } else {
                    settings.selected_content.files.selected = settings.selected_content.files.selected.filter(url => url !== file.url);
                }
                Object.assign(extension_settings.vectors_enhanced, settings);
                saveSettingsDebounced();
            });
            
            fileList.append(checkbox);
        });
    }
    
    if (chatFiles.length > 0) {
        if (dataBankFiles.length > 0) fileList.append('<hr class="m-t-0-5 m-b-0-5">');
        fileList.append('<div class="file-group-header">Chat Attachments</div>');
        chatFiles.forEach(file => {
            const isChecked = settings.selected_content.files.selected.includes(file.url);
            const checkbox = $(`
                <label class="checkbox_label flex-container alignItemsCenter" title="${file.name}">
                    <input type="checkbox" value="${file.url}" ${isChecked ? 'checked' : ''} />
                    <span class="flex1 text-overflow-ellipsis">${file.name} (${(file.size / 1024).toFixed(1)} KB)</span>
                </label>
            `);
            
            checkbox.find('input').on('change', function() {
                if (this.checked) {
                    if (!settings.selected_content.files.selected.includes(file.url)) {
                        settings.selected_content.files.selected.push(file.url);
                    }
                } else {
                    settings.selected_content.files.selected = settings.selected_content.files.selected.filter(url => url !== file.url);
                }
                Object.assign(extension_settings.vectors_enhanced, settings);
                saveSettingsDebounced();
            });
            
            fileList.append(checkbox);
        });
    }
}

/**
 * Updates the World Info list UI
 */
async function updateWorldInfoList() {
    const entries = await getSortedEntries();
    const wiList = $('#vectors_enhanced_wi_list');
    wiList.empty();
    
    if (!entries || entries.length === 0) {
        wiList.append('<div class="text-muted">No available World Info entries</div>');
        return;
    }
    
    // Group entries by world
    const grouped = {};
    entries.forEach(entry => {
        if (!entry.world || entry.disable || !entry.content) return;
        if (!grouped[entry.world]) grouped[entry.world] = [];
        grouped[entry.world].push(entry);
    });
    
    if (Object.keys(grouped).length === 0) {
        wiList.append('<div class="text-muted">No valid World Info entries found</div>');
        return;
    }
    
    for (const [world, worldEntries] of Object.entries(grouped)) {
        const worldDiv = $('<div class="wi-world-group"></div>');
        
        // World name and select-all checkbox
        const selectedEntries = settings.selected_content.world_info.selected[world] || [];
        const allChecked = worldEntries.length > 0 && worldEntries.every(e => selectedEntries.includes(e.uid));
        
        const worldHeader = $(`
            <div class="wi-world-header flex-container alignItemsCenter" style="cursor: pointer;">
                <i class="fa-solid fa-chevron-right wi-world-toggle" style="margin-right: 0.25rem; font-size: 0.8em; color: var(--SmartThemeQuoteColor); transition: transform 0.2s; flex-shrink: 0;"></i>
                <label class="checkbox_label flex1" style="pointer-events: none; display: flex; align-items: center; min-width: 0;">
                    <input type="checkbox" class="world-select-all" data-world="${world}" ${allChecked ? 'checked' : ''} style="pointer-events: auto; margin-right: 0.25rem; flex-shrink: 0;" />
                    <span class="wi-world-name" style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${world}">${world}</span>
                </label>
            </div>
        `);
        
        // Create entry container (hidden by default)
        const entriesContainer = $('<div class="wi-entries-container" style="display: none; padding-left: 2.25rem;"></div>');
        
        // Collapse/expand event
        worldHeader.on('click', function(e) {
            // If the checkbox is clicked, don't handle the collapse event
            if ($(e.target).is('input[type="checkbox"]')) {
                return;
            }
            
            const isExpanded = entriesContainer.is(':visible');
            const toggleIcon = worldHeader.find('.wi-world-toggle');
            
            if (isExpanded) {
                entriesContainer.slideUp(200);
                toggleIcon.css('transform', 'rotate(0deg)');
            } else {
                entriesContainer.slideDown(200);
                toggleIcon.css('transform', 'rotate(90deg)');
            }
        });
        
        // Select-all checkbox event
        worldHeader.find('.world-select-all').on('change', function() {
            const isChecked = this.checked;
            
            if (isChecked) {
                settings.selected_content.world_info.selected[world] = worldEntries.map(e => e.uid);
            } else {
                delete settings.selected_content.world_info.selected[world];
            }
            
            // Update all child entries
            entriesContainer.find('.wi-entry input').prop('checked', isChecked);
            
            Object.assign(extension_settings.vectors_enhanced, settings);
            saveSettingsDebounced();
        });
        
        worldDiv.append(worldHeader);
        
        // Entry list
        worldEntries.forEach(entry => {
            const isChecked = selectedEntries.includes(entry.uid);
            
            const checkbox = $(`
                <label class="checkbox_label wi-entry flex-container alignItemsCenter">
                    <input type="checkbox" value="${entry.uid}" data-world="${world}" ${isChecked ? 'checked' : ''} />
                    <span class="flex1">${entry.comment || '(No comment)'}</span>
                </label>
            `);
            
            checkbox.find('input').on('change', function() {
                if (!settings.selected_content.world_info.selected[world]) {
                    settings.selected_content.world_info.selected[world] = [];
                }
                
                if (this.checked) {
                    if (!settings.selected_content.world_info.selected[world].includes(entry.uid)) {
                        settings.selected_content.world_info.selected[world].push(entry.uid);
                    }
                } else {
                    settings.selected_content.world_info.selected[world] =
                        settings.selected_content.world_info.selected[world].filter(id => id !== entry.uid);
                }
                
                // Update select-all checkbox state
                const allChecked = worldEntries.every(e =>
                    settings.selected_content.world_info.selected[world]?.includes(e.uid)
                );
                worldHeader.find('.world-select-all').prop('checked', allChecked);
                
                // Clean up empty world arrays
                if (settings.selected_content.world_info.selected[world].length === 0) {
                    delete settings.selected_content.world_info.selected[world];
                }
                
                Object.assign(extension_settings.vectors_enhanced, settings);
                saveSettingsDebounced();
            });
            
            entriesContainer.append(checkbox);
        });
        
        worldDiv.append(entriesContainer);
        wiList.append(worldDiv);
    }
}

/**
 * Updates chat message range settings
 */
function updateChatSettings() {
    const context = getContext();
    const messageCount = context.chat?.length || 0;
    
    $('#vectors_enhanced_chat_start').attr('max', messageCount);
    $('#vectors_enhanced_chat_end').attr('min', -1).attr('max', messageCount);
}

// Event handlers
const onChatEvent = debounce(async () => {
    if (settings.auto_vectorize) {
        await moduleWorker.update();
    }
    // Update UI lists when chat changes
    await updateFileList();
    updateChatSettings();
    await updateTaskList();
}, debounce_timeout.relaxed);

/**
 * Debug function: Explore message hiding mechanism
 * @returns {void}
 */
function debugHiddenMessages() {
    const context = getContext();
    if (!context.chat || context.chat.length === 0) {
        console.log('Debug: No chat messages available');
        return;
    }
    
    console.log('=== Start exploring message hiding mechanism ===');
    console.log(`Total messages: ${context.chat.length}`);
    
    // Check the full structure of the first 5 messages
    console.log('\nFull structure of the first 5 messages:');
    context.chat.slice(0, 5).forEach((msg, index) => {
        console.log(`\nMessage #${index}:`, msg);
        
        // Check for possible hidden properties
        const possibleHiddenProps = ['hidden', 'is_hidden', 'hide', 'isHidden', 'visible', 'is_visible'];
        console.log(`Checking for possible hidden properties:`);
        possibleHiddenProps.forEach(prop => {
            if (prop in msg) {
                console.log(`  - ${prop}: ${msg[prop]}`);
            }
        });
        
        // Check extra object
        if (msg.extra) {
            console.log(`  extra object:`, msg.extra);
        }
    });
    
    console.log('\n=== Exploration finished ===');
}

/**
 * Debug function: Test slash command execution
 * @returns {Promise<void>}
 */
async function debugSlashCommands() {
    console.log('=== Testing slash command execution ===');
    
    try {
        // Check possible command execution methods
        const context = getContext();
        console.log('\nChecking context object:', context);
        
        // Method 1: Directly modify the is_system property of a message
        if (context.chat && context.chat.length > 0) {
            console.log('\nTesting direct modification of message property:');
            const testMessage = context.chat[0];
            console.log('is_system status of the first message:', testMessage.is_system);
            console.log('You can hide/show messages by modifying the is_system property');
        }
        
        // Method 2: Check global functions
        const globalFunctions = Object.keys(window).filter(key =>
            key.includes('hide') || key.includes('slash') || key.includes('command')
        );
        console.log('\nRelated global functions:', globalFunctions);
        
        // Method 3: Check jQuery events
        console.log('\nChecking event handlers of message elements...');
        const messageElement = $('.mes').first();
        if (messageElement.length > 0) {
            const events = $._data(messageElement[0], 'events');
            console.log('Events of the message element:', events);
        }
        
    } catch (error) {
        console.error('Error while debugging slash commands:', error);
    }
    
    console.log('=== Testing finished ===');
}

/**
 * Shows a list of currently hidden messages
 * @returns {Promise<void>}
 */
async function showHiddenMessages() {
    const hidden = getHiddenMessages();
    
    if (hidden.length === 0) {
        await callGenericPopup('There are currently no hidden messages', POPUP_TYPE.TEXT, '', { okButton: 'Close' });
        return;
    }
    
    // Calculate the floor ranges of hidden messages
    const indexes = hidden.map(msg => msg.index).sort((a, b) => a - b);
    const ranges = [];
    let start = indexes[0];
    let end = indexes[0];
    
    for (let i = 1; i < indexes.length; i++) {
        if (indexes[i] === end + 1) {
            end = indexes[i];
        } else {
            ranges.push(start === end ? `Floor ${start}` : `Floors ${start}-${end}`);
            start = end = indexes[i];
        }
    }
    ranges.push(start === end ? `Floor ${start}` : `Floors ${start}-${end}`);
    
    const rangeText = ranges.join(', ');
    
    let html = '<div class="hidden-messages-popup">';
    html += `<h3 style="color: var(--SmartThemeQuoteColor); margin-bottom: 1rem; font-size: 1.2em; text-align: left;">Hidden Floors: ${rangeText}</h3>`;
    html += '<div class="hidden-messages-all-content" style="max-height: 60vh; overflow-y: auto; padding: 1rem; background-color: var(--SmartThemeBlurTintColor); border-radius: 6px; text-align: left; white-space: pre-wrap; word-break: break-word;">';
    
    // Sort by index and display all hidden messages
    hidden.sort((a, b) => a.index - b.index).forEach((msg, idx) => {
        const msgType = msg.is_user ? 'User' : 'AI';
        html += `<span style="color: var(--SmartThemeQuoteColor); font-weight: bold;">#${msg.index} - ${msgType} (${msg.name}):</span>\n${msg.text}\n\n`;
    });
    
    html += '</div></div>';
    
    await callGenericPopup(html, POPUP_TYPE.TEXT, '', {
        okButton: 'Close',
        wide: true,
        large: true
    });
}

/**
 * Updates the hidden messages info display
 */
function updateHiddenMessagesInfo() {
    const hidden = getHiddenMessages();
    const infoDiv = $('#vectors_enhanced_hidden_info');
    const countSpan = $('#vectors_enhanced_hidden_count');
    const listDiv = $('#vectors_enhanced_hidden_list');
    
    countSpan.text(hidden.length);
    
    if (hidden.length > 0) {
        infoDiv.show();
        listDiv.empty();
        
        // Only show a preview of the first 5 hidden messages
        const preview = hidden.slice(0, 5);
        preview.forEach(msg => {
            const msgType = msg.is_user ? 'User' : 'AI';
            const item = $(`
                <div class="hidden-message-preview">
                    <strong>#${msg.index}</strong> - ${msgType}: ${msg.text}
                </div>
            `);
            listDiv.append(item);
        });
        
        if (hidden.length > 5) {
            listDiv.append('<div class="text-muted">...and ${hidden.length - 5} more hidden messages</div>');
        }
    } else {
        infoDiv.hide();
    }
}

/**
 * Toggles the hidden state of a message
 * @param {number} messageIndex Message index
 * @param {boolean} hide Whether to hide
 * @returns {Promise<boolean>} Whether successful
 */
async function toggleMessageVisibility(messageIndex, hide) {
    const context = getContext();
    if (!context.chat || messageIndex < 0 || messageIndex >= context.chat.length) {
        console.error('Invalid message index:', messageIndex);
        return false;
    }
    
    try {
        // Modify the is_system property of the message
        context.chat[messageIndex].is_system = hide;
        
        // Trigger save
        await context.saveChat();
        
        // Refresh the UI
        await context.reloadCurrentChat();
        
        return true;
    } catch (error) {
        console.error('Failed to toggle message visibility:', error);
        return false;
    }
}

/**
 * Bulk toggles the hidden state of a message range
 * @param {number} startIndex Start index
 * @param {number} endIndex End index (exclusive)
 * @param {boolean} hide Whether to hide
 * @returns {Promise<void>}
 */
async function toggleMessageRangeVisibility(startIndex, endIndex, hide) {
    const context = getContext();
    if (!context.chat) {
        toastr.error('No chat history available');
        return;
    }
    
    // Ensure indexes are valid
    const actualEnd = endIndex === -1 ? context.chat.length : Math.min(endIndex, context.chat.length);
    const actualStart = Math.max(0, startIndex);
    
    if (actualStart >= actualEnd) {
        toastr.error('Invalid message range');
        return;
    }
    
    try {
        // Bulk modify messages
        let modifiedCount = 0;
        for (let i = actualStart; i < actualEnd; i++) {
            const msg = context.chat[i];
            if (!msg) continue;
            
            // Only process user and AI messages, skip real system messages
            // Note: is_user indicates a user message, messages without this flag are usually AI messages
            // is_system is used to mark the hidden state
            if (msg.is_user !== undefined || msg.name) {
                context.chat[i].is_system = hide;
                modifiedCount++;
            }
        }
        
        // Save and refresh
        await context.saveChat();
        await context.reloadCurrentChat();
        
        const action = hide ? 'Hid' : 'Shown';
        toastr.success(`${action} ${modifiedCount} messages`);
    } catch (error) {
        console.error('Failed to bulk toggle message visibility:', error);
        toastr.error('Operation failed: ' + error.message);
    }
}

/**
 * Gets information about currently hidden messages
 * @returns {Array<{index: number, text: string}>} List of hidden messages
 */
function getHiddenMessages() {
    const context = getContext();
    if (!context.chat) return [];
    
    const hidden = [];
    context.chat.forEach((msg, index) => {
        if (msg.is_system === true) {
            hidden.push({
                index: index,
                text: msg.mes ? msg.mes.substring(0, 100) + (msg.mes.length > 100 ? '...' : '') : '',
                is_user: msg.is_user,
                name: msg.name
            });
        }
    });
    
    return hidden;
}

jQuery(async () => {
    // Use a separate settings key to avoid conflicts
    const SETTINGS_KEY = 'vectors_enhanced';
    
    if (!extension_settings[SETTINGS_KEY]) {
        extension_settings[SETTINGS_KEY] = settings;
    }

    // Deep merge settings to ensure all required properties exist
    Object.assign(settings, extension_settings[SETTINGS_KEY]);
    
    // Ensure chat types exist (for backward compatibility)
    if (!settings.selected_content.chat.types) {
        settings.selected_content.chat.types = { user: true, assistant: true };
    }
    
    // Ensure include_hidden property exists
    if (settings.selected_content.chat.include_hidden === undefined) {
        settings.selected_content.chat.include_hidden = false;
    }
    
    // Ensure all required structures exist
    if (!settings.selected_content.chat.range) {
        settings.selected_content.chat.range = { start: 0, end: -1 };
    }
    
    // Ensure vector_tasks exists
    if (!settings.vector_tasks) {
        settings.vector_tasks = {};
    }
    
    // Save the corrected settings
    Object.assign(extension_settings[SETTINGS_KEY], settings);
    saveSettingsDebounced();

    // Third-party plugins need to use the full path
    const template = await renderExtensionTemplateAsync('third-party/vectors-enhanced', 'settings');
    $('#extensions_settings2').append(template);

    // Initialize master switch first
    $('#vectors_enhanced_master_enabled').prop('checked', settings.master_enabled).on('change', function() {
        settings.master_enabled = $(this).prop('checked');
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
        updateMasterSwitchState();
    });

    // Initialize master switch state
    updateMasterSwitchState();

    // Initialize UI elements
    $('#vectors_enhanced_source').val(settings.source).on('change', () => {
        settings.source = String($('#vectors_enhanced_source').val());
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
        toggleSettings();
    });

    $('#vectors_enhanced_vllm_model').val(settings.vllm_model).on('input', () => {
        settings.vllm_model = String($('#vectors_enhanced_vllm_model').val());
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
    });

    $('#vectors_enhanced_vllm_url').val(settings.vllm_url).on('input', () => {
        settings.vllm_url = String($('#vectors_enhanced_vllm_url').val());
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
    });

    $('#vectors_enhanced_local_model').val(settings.local_model).on('input', () => {
        settings.local_model = String($('#vectors_enhanced_local_model').val());
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
    });

    $('#vectors_enhanced_ollama_model').val(settings.ollama_model).on('input', () => {
        settings.ollama_model = String($('#vectors_enhanced_ollama_model').val());
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
    });

    $('#vectors_enhanced_ollama_url').val(settings.ollama_url).on('input', () => {
        settings.ollama_url = String($('#vectors_enhanced_ollama_url').val());
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
    });

    $('#vectors_enhanced_auto_vectorize').prop('checked', settings.auto_vectorize).on('input', () => {
        settings.auto_vectorize = $('#vectors_enhanced_auto_vectorize').prop('checked');
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
    });

    $('#vectors_enhanced_chunk_size').val(settings.chunk_size).on('input', () => {
        settings.chunk_size = Number($('#vectors_enhanced_chunk_size').val());
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
    });

    $('#vectors_enhanced_overlap_percent').val(settings.overlap_percent).on('input', () => {
        settings.overlap_percent = Number($('#vectors_enhanced_overlap_percent').val());
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
    });

    $('#vectors_enhanced_score_threshold').val(settings.score_threshold).on('input', () => {
        settings.score_threshold = Number($('#vectors_enhanced_score_threshold').val());
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
    });

    $('#vectors_enhanced_force_chunk_delimiter').val(settings.force_chunk_delimiter).on('input', () => {
        settings.force_chunk_delimiter = String($('#vectors_enhanced_force_chunk_delimiter').val());
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
    });

    $('#vectors_enhanced_enabled').prop('checked', settings.enabled).on('input', () => {
        settings.enabled = $('#vectors_enhanced_enabled').prop('checked');
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
    });

    $('#vectors_enhanced_query_messages').val(settings.query_messages).on('input', () => {
        settings.query_messages = Number($('#vectors_enhanced_query_messages').val());
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
    });

    $('#vectors_enhanced_max_results').val(settings.max_results).on('input', () => {
        settings.max_results = Number($('#vectors_enhanced_max_results').val());
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
    });
    
    // Content tag settings event handlers
    $('#vectors_enhanced_tag_chat').on('input', () => {
        const value = $('#vectors_enhanced_tag_chat').val().trim() || 'past_chat';
        settings.content_tags.chat = value;
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
    });
    
    $('#vectors_enhanced_tag_wi').on('input', () => {
        const value = $('#vectors_enhanced_tag_wi').val().trim() || 'world_part';
        settings.content_tags.world_info = value;
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
    });
    
    $('#vectors_enhanced_tag_file').on('input', () => {
        const value = $('#vectors_enhanced_tag_file').val().trim() || 'databank';
        settings.content_tags.file = value;
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
    });

    $('#vectors_enhanced_template').val(settings.template).on('input', () => {
        settings.template = String($('#vectors_enhanced_template').val());
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
    });

    $('#vectors_enhanced_depth').val(settings.depth).on('input', () => {
        settings.depth = Number($('#vectors_enhanced_depth').val());
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
    });

    $(`input[name="vectors_position"][value="${settings.position}"]`).prop('checked', true);
    $('input[name="vectors_position"]').on('change', () => {
        settings.position = Number($('input[name="vectors_position"]:checked').val());
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
    });

    $('#vectors_enhanced_depth_role').val(settings.depth_role).on('change', () => {
        settings.depth_role = Number($('#vectors_enhanced_depth_role').val());
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
    });

    $('#vectors_enhanced_include_wi').prop('checked', settings.include_wi).on('input', () => {
        settings.include_wi = $('#vectors_enhanced_include_wi').prop('checked');
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
    });

    // Content selection handlers
    $('#vectors_enhanced_chat_enabled').prop('checked', settings.selected_content.chat.enabled).on('input', () => {
        settings.selected_content.chat.enabled = $('#vectors_enhanced_chat_enabled').prop('checked');
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
        updateContentSelection();
    });

    $('#vectors_enhanced_files_enabled').prop('checked', settings.selected_content.files.enabled).on('input', async () => {
        settings.selected_content.files.enabled = $('#vectors_enhanced_files_enabled').prop('checked');
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
        updateContentSelection();
        if (settings.selected_content.files.enabled) {
            await updateFileList();
        }
    });

    $('#vectors_enhanced_wi_enabled').prop('checked', settings.selected_content.world_info.enabled).on('input', async () => {
        settings.selected_content.world_info.enabled = $('#vectors_enhanced_wi_enabled').prop('checked');
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
        updateContentSelection();
        if (settings.selected_content.world_info.enabled) {
            await updateWorldInfoList();
        }
    });

    // Chat settings handlers - ensure all properties exist
    const chatRange = settings.selected_content.chat.range || { start: 0, end: -1 };
    const chatTypes = settings.selected_content.chat.types || { user: true, assistant: true };
    const chatTags = settings.selected_content.chat.tags || '';
    const includeHidden = settings.selected_content.chat.include_hidden || false;
    
    $('#vectors_enhanced_chat_start').val(chatRange.start).on('input', () => {
        if (!settings.selected_content.chat.range) {
            settings.selected_content.chat.range = { start: 0, end: -1 };
        }
        settings.selected_content.chat.range.start = Number($('#vectors_enhanced_chat_start').val());
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
    });

    $('#vectors_enhanced_chat_end').val(chatRange.end).on('input', () => {
        if (!settings.selected_content.chat.range) {
            settings.selected_content.chat.range = { start: 0, end: -1 };
        }
        settings.selected_content.chat.range.end = Number($('#vectors_enhanced_chat_end').val());
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
    });

    // Message type checkboxes
    $('#vectors_enhanced_chat_user').prop('checked', chatTypes.user).on('input', () => {
        if (!settings.selected_content.chat.types) {
            settings.selected_content.chat.types = { user: true, assistant: true };
        }
        settings.selected_content.chat.types.user = $('#vectors_enhanced_chat_user').prop('checked');
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
    });

    $('#vectors_enhanced_chat_assistant').prop('checked', chatTypes.assistant).on('input', () => {
        if (!settings.selected_content.chat.types) {
            settings.selected_content.chat.types = { user: true, assistant: true };
        }
        settings.selected_content.chat.types.assistant = $('#vectors_enhanced_chat_assistant').prop('checked');
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
    });

    $('#vectors_enhanced_chat_tags').val(chatTags).on('input', () => {
        settings.selected_content.chat.tags = String($('#vectors_enhanced_chat_tags').val());
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
    });
    
    $('#vectors_enhanced_include_hidden').prop('checked', includeHidden).on('input', () => {
        settings.selected_content.chat.include_hidden = $('#vectors_enhanced_include_hidden').prop('checked');
        Object.assign(extension_settings.vectors_enhanced, settings);
        saveSettingsDebounced();
    });

    // Button handlers
    $('#vectors_enhanced_hide_range').on('click', async () => {
        const start = Number($('#vectors_enhanced_chat_start').val());
        const end = Number($('#vectors_enhanced_chat_end').val());
        await toggleMessageRangeVisibility(start, end, true);
    });

    $('#vectors_enhanced_unhide_range').on('click', async () => {
        const start = Number($('#vectors_enhanced_chat_start').val());
        const end = Number($('#vectors_enhanced_chat_end').val());
        await toggleMessageRangeVisibility(start, end, false);
    });

    $('#vectors_enhanced_show_hidden').on('click', showHiddenMessages);

    $('#vectors_enhanced_files_refresh').on('click', updateFileList);
    $('#vectors_enhanced_wi_refresh').on('click', updateWorldInfoList);

    // Initial UI setup
    toggleSettings();
    updateContentSelection();
    updateChatSettings();
    updateFileList();
    updateWorldInfoList();
    updateTaskList();
    updateHiddenMessagesInfo();

    // Event listeners
    eventSource.on(event_types.chatChanged, onChatEvent);
    eventSource.on(event_types.characterChanged, onChatEvent);
    eventSource.on(event_types.messageDeleted, onChatEvent);
    eventSource.on(event_types.messageEdited, onChatEvent);
    eventSource.on(event_types.messageSwiped, onChatEvent);
    
    // Register debug functions
    registerDebugFunction('vectors_debug_hidden', debugHiddenMessages);
    registerDebugFunction('vectors_debug_commands', debugSlashCommands);
});