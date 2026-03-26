// Global state
let currentGeneratedText = '';
let currentAttentionData = null;

// Auto-resize textarea
function autoResizeTextarea(textarea) {
    textarea.style.height = '2.8rem'; // Reset to initial height (matches new padding)
    if (textarea.scrollHeight > textarea.clientHeight) {
        textarea.style.height = textarea.scrollHeight + 'px';
    }
}

// Initialize auto-resize for textareas
window.addEventListener('DOMContentLoaded', () => {
    const textareas = document.querySelectorAll('textarea');
    textareas.forEach(textarea => {
        textarea.addEventListener('input', function() {
            autoResizeTextarea(this);
        });
        // Don't initialize height on load - let CSS handle it
    });
});

// Tab switching
function switchTab(tabName) {
    // Update tab buttons
    document.querySelectorAll('.tab').forEach(tab => {
        tab.classList.remove('active');
    });
    event.target.classList.add('active');

    // Update tab content
    document.querySelectorAll('.tab-content').forEach(content => {
        content.classList.remove('active');
    });
    document.getElementById(`${tabName}-tab`).classList.add('active');
}

// Generate text
async function generateText() {
    const prompt = document.getElementById('prompt').value.trim();
    const temperature = parseFloat(document.getElementById('temperature').value);
    const maxTokens = parseInt(document.getElementById('max-tokens').value);

    if (!prompt) {
        showError('generate-error', 'Please enter a prompt');
        return;
    }

    hideError('generate-error');
    document.getElementById('generate-loading').classList.add('active');
    document.getElementById('generate-output-card').style.display = 'none';

    try {
        const data = await callGenerateAPI(prompt, temperature, maxTokens);
        currentGeneratedText = data.answer;

        document.getElementById('generated-text').textContent = data.answer;

        const metadata = data.metadata;
        document.getElementById('generate-metadata').innerHTML = `
            <div class="metadata-item"><strong>Model:</strong> ${data.model}</div>
            <div class="metadata-item"><strong>Temperature:</strong> ${metadata.temperature}</div>
            <div class="metadata-item"><strong>Max New Tokens:</strong> ${metadata.max_tokens}</div>
            <div class="metadata-item"><strong>Prompt Tokens:</strong> ${metadata.prompt_tokens}</div>
            <div class="metadata-item"><strong>Generated Tokens:</strong> ${metadata.generated_tokens}</div>
        `;

        document.getElementById('generate-output-card').style.display = 'block';
    } catch (error) {
        showError('generate-error', `Error: ${error.message}`);
    } finally {
        document.getElementById('generate-loading').classList.remove('active');
    }
}

// Use generated text in visualization tab
function useGeneratedText() {
    const analyzeTextarea = document.getElementById('analyze-text');
    analyzeTextarea.value = currentGeneratedText;

    // Switch tabs first so the textarea is visible
    document.querySelectorAll('.tab').forEach((tab, index) => {
        tab.classList.remove('active');
        if (index === 1) tab.classList.add('active');
    });
    document.querySelectorAll('.tab-content').forEach((content, index) => {
        content.classList.remove('active');
        if (index === 1) content.classList.add('active');
    });

    // Now resize after the textarea is visible
    setTimeout(() => {
        autoResizeTextarea(analyzeTextarea);
    }, 0);
}

// Analyze text
async function analyzeText() {
    const text = document.getElementById('analyze-text').value.trim();
    const attnLayer = parseInt(document.getElementById('attn-layer').value);

    if (!text) {
        showError('visualize-error', 'Please enter text to analyze');
        return;
    }

    hideError('visualize-error');
    document.getElementById('analyze-loading').classList.add('active');
    document.getElementById('analyze-output-card').style.display = 'none';

    try {
        const data = await callAnalyzeAPI(text, attnLayer);
        currentAttentionData = data;

        const numLayers = data.shape[0];
        const numHeads = data.shape[1];
        const seqLen = data.shape[2];

        document.getElementById('analyze-metadata').innerHTML = `
            <div class="metadata-item"><strong>Shape:</strong> [${data.shape.join(', ')}]</div>
            <div class="metadata-item"><strong>Layers:</strong> ${numLayers}</div>
            <div class="metadata-item"><strong>Heads:</strong> ${numHeads}</div>
            <div class="metadata-item"><strong>Sequence Length:</strong> ${seqLen}</div>
            <div class="metadata-item"><strong>Tokens:</strong> ${data.num_tokens}</div>
        `;

        document.getElementById('analyze-output-card').style.display = 'block';

        // Initialize visualization
        if (typeof initializeVisualization === 'function') {
            initializeVisualization(currentAttentionData);
        }
    } catch (error) {
        showError('visualize-error', `Error: ${error.message}`);
    } finally {
        document.getElementById('analyze-loading').classList.remove('active');
    }
}

// Utility functions
function showError(elementId, message) {
    const errorElement = document.getElementById(elementId);
    errorElement.textContent = message;
    errorElement.classList.add('active');
}

function hideError(elementId) {
    const errorElement = document.getElementById(elementId);
    errorElement.classList.remove('active');
}
