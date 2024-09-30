<template>
  <div id="app" class="bg-dark text-white">
    <div class="container text-center mb-4 pt-4">
      <h1 v-if='!done'>Tag this image</h1>
      <h1 v-else>You're All Done!</h1>
    </div>

    <div class='d-flex flex-row justify-content-stretch align-content-stretch align-items-stretch' style='padding:4px;'>
        <div :style="{visibility: currentIndex > 0 ? 'visible' : 'hidden' }" @click="prevImage" class='left-right-button left-button'><i class="bi bi-arrow-left-circle"></i></div>

        <div v-if='!done' class='image-container p-4 rounded'>
          <img :src="currentImageSrc" class="rounded main-image"/>
          <div class='x-of-y'>{{currentIndex+1}} / {{images.length}}</div>
        </div>
        <div v-else @click="exitServer" class='image-container p-4 rounded d-flex flex-row justify-content-center align-items-center continue-button'>
          <b>Click to Continue</b>
        </div>

        <div :style="{ visibility: !done ? 'visible' : 'hidden' }" @click="nextImage" class='left-right-button right-button'><i class="bi bi-arrow-right-circle"></i></div>
    </div>

    <div class='d-flex flex-column justify-content-center'>
      <div v-for='tag_row in tag_layout' class='d-flex flex-row flex-wrap justify-content-center'>
        <div v-for='tag in tag_row' :class="[currentImage.tags.includes(tag) ? 'tag-active' : '', 'btn btn-secondary m-1 tag']" @click='tagClicked(tag)'>
          {{tag}}
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
#app {
  min-height: 100vh;
}
.tag {
  display: inline-block;
  padding: 0.2em 0.4em;
  font-size: 2.0em;
  color: #fff;
  background-color: #6c757d;
  border-radius: 0.2em;
  border: 1px solid #fff;
  margin-right: 4px;
  touch-action: manipulation;
}
.tag-active {
  background-color: #007bff;
  color: #fff;
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
.continue-button {
  cursor: pointer;
  font-size: 200%;
}
.main-image {
  width: 100%; 
  height: 100%; 
  object-fit: contain
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
const tag_layout = ref([]);
const currentIndex = ref(0);
const done = ref(false);

const currentImage = computed(() => images.value[currentIndex.value]);
const currentImageSrc = computed(() => currentImage.value ? `${baseUrl}/images/${currentImage.value.id}` : '');

const tagClicked = (tag) => {
  if (currentImage.value.tags.includes(tag)) {
    currentImage.value.tags = currentImage.value.tags.filter(t => t !== tag);
  } else {
    currentImage.value.tags.push(tag);
  }
  backend.post(`/api/images/${currentImage.value.id}`, { tags: currentImage.value.tags });
};

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

const loadTags = async () => {
  try {
    const response = await backend.get('/api/tags');
    tag_layout.value = response.data;
  } catch (error) {
    console.error('Failed to load tags:', error);
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
  loadTags();
  window.addEventListener('keydown', handleKeydown);
});

onBeforeUnmount(() => {
  window.removeEventListener('keydown', handleKeydown);
});
</script>
