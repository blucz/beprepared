<template>
  <div id="app" class="bg-dark text-white h-100 w-100">
    <div class="container text-center mb-4 pt-4">
      <h1>Generic Applet</h1>
    </div>

    <!-- spinner -->
    <div class="spinner-border" role="status">
      <span class="visually-hidden">Loading...</span>
    </div>
  </div>
</template>

<style scoped>
</style>

<script setup>
import { ref, computed, onMounted, onBeforeUnmount } from 'vue';
import axios from 'axios';

console.log(import.meta.env);

const props = defineProps({
  apiPath: { type: String, required: true },
});
const emit = defineEmits(['close']);

const baseURL = (import.meta.env.VITE_API_URL ?? '') + (props.apiPath??'');
const backend = axios.create({ baseURL })

</script>
