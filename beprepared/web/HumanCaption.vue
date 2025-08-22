<template>
  <div id="app" class="bg-dark text-white">
    <div class="header-bar">
      <div class="header-content">
        <h1>Edit caption for this image</h1>
        <div class="progress-indicator">{{currentIndex+1}} / {{images.length}}</div>
      </div>
    </div>

    <div class='content-wrapper' v-if='!done'>
      <div class='main-content'>
        <!-- Navigation buttons -->
        <div :style="{visibility: canGoPrevious ? 'visible' : 'hidden' }" @click="prevImage" class='nav-button nav-left'>
          <i class="bi bi-arrow-left-circle"></i>
        </div>

        <!-- Image section -->
        <div class='image-section'>
          <img :src="currentImageSrc" class="main-image"/>
        </div>

        <!-- Caption section -->
        <div class='caption-section' v-if='!exited && currentImage'>
          <label class='caption-label'>Caption:</label>
          <textarea 
            v-model="currentImage.caption" 
            @input="updateCaption"
            :disabled='done'
            class='caption-textarea'
            placeholder="Enter caption for this image...">
          </textarea>
        </div>

        <!-- Navigation buttons -->
        <div :style="{ visibility: canGoNext ? 'visible' : 'hidden' }" @click="nextImage" class='nav-button nav-right'>
          <i class="bi bi-arrow-right-circle"></i>
        </div>
      </div>
      
      <!-- Keyboard shortcuts below -->
      <div class="shortcuts-bar hide-on-touch">
        <div class="shortcuts-info">
          <div class="shortcut-item">
            <kbd>←</kbd> <kbd>A</kbd> <span>Previous Image</span>
          </div>
          <div class="shortcut-item">
            <kbd>→</kbd> <kbd>D</kbd> <span>Next Image</span>
          </div>
          <div class="shortcut-item">
            <kbd>Enter</kbd> <span>Continue (when done)</span>
          </div>
        </div>
      </div>
    </div>

    <!-- Done state -->
    <div v-else class='done-container'>
      <button v-if='!exited' class='btn btn-primary btn-lg' @click='close'>Continue to next step</button>
    </div>
  </div>
</template>

<style scoped>
#app {
  min-height: 100vh;
  display: flex;
  flex-direction: column;
}

/* Header */
.header-bar {
  background-color: #2d3338;
  padding: 1rem 2rem;
  border-bottom: 1px solid #6c757d;
}

.header-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  max-width: 1400px;
  margin: 0 auto;
}

.header-content h1 {
  margin: 0;
  font-size: 1.5rem;
}

.progress-indicator {
  font-size: 1.2rem;
  font-weight: bold;
  color: #8b949e;
}

/* Content wrapper */
.content-wrapper {
  flex: 1;
  display: flex;
  flex-direction: column;
}

/* Main content area */
.main-content {
  flex: 1;
  display: grid;
  grid-template-columns: auto 1fr 1fr auto;
  align-items: stretch;
  padding: 20px;
  gap: 20px;
  max-width: 1600px;
  margin: 0 auto;
  width: 100%;
  height: calc(100vh - 160px); /* Fixed height for header and shortcuts bar */
  min-height: 80vh; /* Minimum 80% of viewport height for landscape images */
}

/* Navigation buttons */
.nav-button {
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
  font-size: 2.5em;
  color: #6c757d;
  padding: 0 20px;
  touch-action: manipulation;
  transition: color 0.2s;
}

.nav-button:hover {
  color: #fff;
}

/* Image section */
.image-section {
  display: flex;
  align-items: center;
  justify-content: center;
  background-color: #343a40;
  border: 1px solid #6c757d;
  border-radius: 0.5rem;
  padding: 20px;
  min-width: 0;
  height: 100%; /* Explicit height */
  min-height: 80vh; /* Ensure minimum height for landscape images */
}

.main-image {
  max-width: 100%;
  max-height: 100%;
  object-fit: contain;
  border-radius: 0.25rem;
}

/* Caption section */
.caption-section {
  display: flex;
  flex-direction: column;
  gap: 10px;
  min-width: 400px;
  max-width: 600px;
  height: 100%; /* Explicit height to match image section */
  min-height: 80vh; /* Ensure minimum height matches image section */
}

.caption-label {
  font-size: 1.1rem;
  font-weight: 500;
  color: #adb5bd;
  flex-shrink: 0; /* Don't shrink */
  height: auto; /* Natural height */
}

.caption-textarea {
  flex: 1;
  width: 100%;
  padding: 15px;
  font-size: 1.1rem;
  line-height: 1.6;
  background-color: #343a40;
  color: #fff;
  border: 1px solid #6c757d;
  border-radius: 0.25rem;
  resize: none; /* Disable resize since it's full height */
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
  overflow-y: auto; /* Add scrollbar if needed */
  box-sizing: border-box; /* Include padding in height */
}

.caption-textarea:focus {
  outline: none;
  border-color: #007bff;
  box-shadow: 0 0 0 0.2rem rgba(0, 123, 255, 0.25);
}

/* Shortcuts bar */
.shortcuts-bar {
  background-color: #2d3338;
  border-top: 1px solid #6c757d;
  padding: 10px 0;
}

.shortcuts-info {
  display: flex;
  justify-content: center;
  gap: 30px;
  align-items: center;
}

.shortcut-item {
  display: flex;
  align-items: center;
  font-size: 0.9rem;
}

.shortcut-item kbd {
  display: inline-block;
  padding: 3px 6px;
  font-size: 0.8rem;
  font-weight: bold;
  line-height: 1;
  color: #fff;
  background-color: #6c757d;
  border: 1px solid #495057;
  border-radius: 3px;
  margin-right: 5px;
}

.shortcut-item span {
  color: #adb5bd;
  margin-left: 5px;
}

/* Done state */
.done-container {
  flex: 1;
  display: flex;
  justify-content: center;
  align-items: center;
  min-height: 400px;
}

.done-container button {
  font-size: 1.5rem;
  padding: 15px 30px;
}

/* Responsive design */
@media (max-width: 1200px) {
  .main-content {
    flex-direction: column;
    height: auto;
  }
  
  .caption-section {
    max-width: 100%;
    min-width: 0;
    height: 300px;
  }
  
  .nav-button {
    position: fixed;
    top: 50%;
    transform: translateY(-50%);
    background-color: rgba(52, 58, 64, 0.9);
    border-radius: 50%;
    width: 60px;
    height: 60px;
    padding: 0;
    z-index: 10;
  }
  
  .nav-left {
    left: 10px;
  }
  
  .nav-right {
    right: 10px;
  }
  
  .caption-textarea {
    resize: vertical; /* Allow resizing on mobile */
  }
}

@media (pointer: coarse) {
  .hide-on-touch {
    display: none;
  }
}
</style>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount, watch } from 'vue';
import axios from 'axios';

const props = defineProps({
  apiPath: { type: String, required: true }
});
const emit = defineEmits(['close']);

const baseURL = (import.meta.env.VITE_API_URL ?? '') + (props.apiPath??'');
const backend = axios.create({ baseURL })

const images = ref([]);
const currentIndex = ref(0);
const done = ref(false);
const exited = ref(false);
const saveTimeout = ref(null);

const currentImage = computed(() => images.value[currentIndex.value]);
const currentImageSrc = computed(() => currentImage.value ? `${baseURL}/objects/${currentImage.value.objectid}` : '');
const canGoPrevious = computed(() => currentIndex.value > 0 && !exited.value);
const canGoNext = computed(() => 
  !done.value && 
  !exited.value && 
  currentIndex.value <= images.value.length - 1);

const showContinueButton = computed(() => done.value && !exited.value);

const saveCaption = async (image) => {
  if (!image) return;
  try {
    const response = await backend.post(`/api/images/${image.id}`, { caption: image.caption });
    if (response.data.status === 'done') {
      done.value = true;
    }
  } catch (error) {
    console.error('Failed to save caption:', error);
  }
};

const updateCaption = () => {
  // Clear existing timeout
  if (saveTimeout.value) {
    clearTimeout(saveTimeout.value);
  }
  // Set new timeout to auto-save after 500ms of no typing
  saveTimeout.value = setTimeout(() => {
    saveCaption(currentImage.value);
  }, 500);
};

const loadImages = async () => {
  try {
    const response = await backend.get('/api/images');
    images.value = response.data.images;
    currentIndex.value = response.data.start_index;
    if (images.value.length === 0) {
      done.value = true;
    }
  } catch (error) {
    console.error('Failed to load images:', error);
  }
};

const prevImage = async () => {
  if (!canGoPrevious.value) return;
  
  // Save current caption before moving
  if (currentImage.value) {
    await saveCaption(currentImage.value);
  }
  
  if (done.value) {
    done.value = false;
  } else if (currentIndex.value > 0) {
    currentIndex.value--;
  }
  preload();
};

const nextImage = async () => {
  if (!canGoNext.value) return;
  
  // Save current caption before moving
  if (currentImage.value) {
    await saveCaption(currentImage.value);
  }
  
  if (currentIndex.value < images.value.length - 1) {
    currentIndex.value++;
  } else {
    done.value = true;
  }
  preload();
};

const preload = () => {
  if (currentIndex.value < images.value.length - 1) {
    const img = new Image();
    img.src = `${baseURL}/objects/${images.value[currentIndex.value + 1].objectid}`;
  }
};

const close = async () => {
  // Save any pending changes
  if (currentImage.value && !done.value) {
    await saveCaption(currentImage.value);
  }
  console.log("close")
  exited.value = true;
  emit('close');
};

const handleKeydown = (event) => {
  // Don't handle shortcuts when textarea is focused and user is typing
  if (event.target.tagName === 'TEXTAREA' && !['ArrowLeft', 'ArrowRight', 'Enter'].includes(event.key)) {
    return;
  }
  
  // Prevent default behavior for navigation keys
  if (['ArrowLeft', 'ArrowRight'].includes(event.key)) {
    event.preventDefault();
  }
  
  switch (event.key) {
    case 'ArrowLeft':
      prevImage();
      break;
    case 'a':
    case 'A':
      if (event.target.tagName !== 'TEXTAREA') {
        prevImage();
      }
      break;
    case 'ArrowRight':
      nextImage();
      break;
    case 'd':
    case 'D':
      if (event.target.tagName !== 'TEXTAREA') {
        nextImage();
      }
      break;
    case 'Enter':
      if (showContinueButton.value && event.target.tagName !== 'TEXTAREA') {
        close();
      }
      break;
  }
};

// Watch for page unload to save current caption
const handleBeforeUnload = async (event) => {
  if (currentImage.value && !exited.value) {
    await saveCaption(currentImage.value);
  }
};

onMounted(() => {
  loadImages();
  window.addEventListener('keydown', handleKeydown);
  window.addEventListener('beforeunload', handleBeforeUnload);
});

onBeforeUnmount(() => {
  // Save any pending changes
  if (currentImage.value && !exited.value) {
    saveCaption(currentImage.value);
  }
  window.removeEventListener('keydown', handleKeydown);
  window.removeEventListener('beforeunload', handleBeforeUnload);
  if (saveTimeout.value) {
    clearTimeout(saveTimeout.value);
  }
});
</script>