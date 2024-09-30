<template>
  <div id="app" class="bg-dark text-white">
    <div class="container text-center mb-4 pt-4">
      <h1>Want to keep this image?</h1>
    </div>

    <div class='d-flex flex-row justify-content-stretch align-content-stretch align-items-stretch' style='padding:4px;'>
        <div :style="{visibility: currentIndex > 0 ? 'visible' : 'hidden' }" @click="prevImage" class='left-right-button left-button'><i class="bi bi-arrow-left-circle"></i></div>

        <div v-if='!done' class='image-container p-4 rounded'>
          <img :src="currentImageSrc" class="rounded" :style="imageBorderStyle"/>
          <div class='x-of-y'>{{currentIndex+1}} / {{images.length}}</div>
        </div>
        <div v-else @click="exitServer" class='image-container p-4 rounded d-flex flex-row justify-content-center align-items-center continue-button'>
          <b>Click to Continue</b>
        </div>

        <div :style="{ visibility: !done ? 'visible' : 'hidden' }" @click="nextImage" class='left-right-button right-button'><i class="bi bi-arrow-right-circle"></i></div>

    </div>

    <div class="d-flex justify-content-center mb-3 flex-row pt-2">
      <button :disabled='done' @click="acceptImage" class="btn btn-success me-2 accept-reject-button">Accept</button>
      <button :disabled='done' @click="rejectImage" class="btn btn-danger accept-reject-button">Reject</button>
    </div>

    <div class="d-flex justify-content-center align-items-center flex-column hide-on-touch">
      <div class="d-flex align-items-left flex-column">
        <div class="mb-1">
          Keyboard shortcuts
        </div>
        <div class="mb-1">
          <div class="keyboard-key">Up</div> <div class="keyboard-key">W</div> <div class='keyboard-key-action'>Accept Image</div>
        </div>
        <div class="mb-1">
            <div class="keyboard-key">Down</div> <div class="keyboard-key">S</div> <div class='keyboard-key-action'>Reject Image</div>
        </div>
        <div class="mb-1">
            <div class="keyboard-key">Left</div> <div class="keyboard-key">A</div> <div class='keyboard-key-action'>Previous Image</div>
        </div>
        <div class="mb-1">
            <div class="keyboard-key">Right</div> <div class="keyboard-key">D</div> <div class='keyboard-key-action'>Next Image</div>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
#app {
  min-height: 100vh;
}
.keyboard-key {
  display: inline-block;
  padding: 0.2em 0.4em;
  font-size: 50%;
  color: #fff;
  background-color: #6c757d;
  border-radius: 0.2em;
  border: 1px solid #fff;
  margin-right: 4px;
  width: 30px;
}
.keyboard-key-action {
  display: inline-block;
  text-align: left;
  font-size: 70%;
}
.left-right-button {
  flex-grow: 1;
  cursor: pointer;
  font-size: 2em;
  display: flex;
  align-items: center;
  color: #6c757d;
  touch-action: manipulation;
}
.left-button {
  justify-content: right;
  padding: 20px;
}
.right-button {
  justify-content: left;
  padding: 20px;
}
.left-right-button:hover {
  color: #fff;
}
.accept-reject-button {
  width: calc(min(800px, 100vw - 200px) / 2.0 - 4px); 
  height: 80px;
  font-size: 150%;
  touch-action: manipulation;
}
.continue-button {
  cursor: pointer;
  font-size: 200%;
}
.image-container {
  background-color: #343a40;
  border: 1px solid #6c757d;
  flex-grow: 0;
  flex-shrink: 0;
  width: calc(min(800px, 100vw - 200px)); 
  height: calc(min(800px, 100vw - 200px)); 
  max-width: 800px;
}
.x-of-y {
  text-align: center;
}
@media (pointer: coarse) {
  .hide-on-touch {
    visibility:hidden;
  }
}
</style>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from 'vue';
import axios from 'axios';

console.log(import.meta.env);

const baseUrl = import.meta.env.VITE_API_URL ?? '';
const backend = baseUrl ? axios.create({ baseURL: import.meta.env.VITE_API_URL })
                        : axios.create();

const images = ref([]);
const currentIndex = ref(0);
const imageStatuses = ref({}); // { id: 'accepted' | 'rejected' }
const done = ref(false);

const currentImage = computed(() => images.value[currentIndex.value]);
const currentImageSrc = computed(() => currentImage.value ? `${baseUrl}/images/${currentImage.value.id}` : '');

const imageBorderStyle = computed(() => {
  const status = imageStatuses.value[currentImage.value?.id];
  if (status === 'accepted') {
    return 'border: 4px solid green; width: 100%; height: 100%; max-width: 800px; max-height: 800px; object-fit: contain';
  } else if (status === 'rejected') {
    return 'border: 4px solid red; width: 100%; height: 100%; max-width: 800px; max-height: 800px; object-fit: contain';
  }
  return 'border: 4px solid transparent; width: 100%; height: 100%; max-width: 800px; max-height: 800px; object-fit: contain';
});

const loadImages = async () => {
  try {
    const response = await backend.get('/api/images');
    images.value = response.data;
    if (images.value.length === 0) {
      done.value = true;
    }
  } catch (error) {
    console.error('Failed to load images:', error);
  }
};

const acceptImage = async () => {
  if (!currentImage.value) return;
  try {
    backend.post(`/api/images/${currentImage.value.id}`, { action: 'accept' });
    imageStatuses.value[currentImage.value.id] = 'accepted';
    nextImage();
  } catch (error) {
    console.error('Failed to accept image:', error);
  }
};

const rejectImage = async () => {
  if (!currentImage.value) return;
  try {
    backend.post(`/api/images/${currentImage.value.id}`, { action: 'reject' });
    imageStatuses.value[currentImage.value.id] = 'rejected';
    nextImage();
  } catch (error) {
    console.error('Failed to reject image:', error);
  }
};

const prevImage = () => {
  if (done.value) {
    done.value = false;
  } else if (currentIndex.value > 0) {
    currentIndex.value--;
  }
  preload();
};

const nextImage = () => {
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
    img.src = `${baseUrl}/images/${images.value[currentIndex.value + 1].id}`;
  }
};

const exitServer = async () => {
  await backend.post('/api/exit');
};

const handleKeydown = (event) => {
  switch (event.key) {
    case 'ArrowUp':
    case 'w':
    case 'W':
      acceptImage();
      break;
    case 'ArrowDown':
    case 's':
    case 'S':
      rejectImage();
      break;
    case 'ArrowLeft':
    case 'a':
    case 'A':
      prevImage();
      break;
    case 'ArrowRight':
    case 'd':
    case 'D':
      nextImage();
      break;
  }
};

onMounted(() => {
  loadImages();
  window.addEventListener('keydown', handleKeydown);
});

onBeforeUnmount(() => {
  window.removeEventListener('keydown', handleKeydown);
});
</script>
