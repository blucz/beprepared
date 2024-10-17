<template>
  <div id="app" class="bg-dark text-white d-flex flex-column vh-100">
      <!-- Header -->
    <header class="header p-2">
      <img src='/beprepared.jpg' alt='beprepared' height='60'/>
    </header>

    <!-- Main content area, which expands to fill available space -->
    <main class="flex-grow-1 d-flex justify-content-center align-items-center bg-secondary">
      <div v-if='!isConnected'>
        <h2>beprepared!</h2>
      </div>

      <div v-if='isConnected && !applet'>
        <div v-if='!logger_name' class="spinner-border" role="status">
          <span class="visually-hidden">Loading...</span>
        </div>
        <div v-else>
          <h2>{{ logger_name }}</h2>
          <div v-for='entry in log' :key='entry.id'>
            <div>{{ entry.message }}</div>
          </div>
        </div>
      </div>

      <component :is='applet' class='flex-grow-1 h-100' 
        v-bind="appletProps" @close='close'
        :style="{ display: applet ? 'block' : 'none' }" />
    </main>

    <footer class="bg-dark text-white p-2 footer">
      <!--Footer content goes here-->
    </footer>
  </div>
</template>

<style scoped>
#app {
  min-height: 100vh;
  min-width: 100vw;
}
.header {
  background: #354044;
}
.footer {
  background: #354044;
}
</style>

<script setup>
import { ref, shallowRef, computed, onMounted, onBeforeUnmount, createVNode, render } from 'vue';
import { createApp, defineAsyncComponent, markRaw } from 'vue';
import axios from 'axios';

console.log(import.meta.env);

const baseUrl = import.meta.env.VITE_API_URL ?? '';
const backend = baseUrl ? axios.create({ baseURL: import.meta.env.VITE_API_URL })
                        : axios.create();
const webSocketUrl = baseUrl ? `ws://${baseUrl.replace('http://','')}/ws` : `ws://${window.location.host}/ws`;
const isConnected = ref(false);
const applet = shallowRef(null);
const appletProps = ref({});
const log = ref([]);
const logger_name = ref(null);

let shouldBeConnected = false;
let websocket;
let websocketSeq = 0;

const activate = async (appletName, path, component) => {
  console.log(`Activating applet ${appletName} ${path} ${component}`);
  const apiPath = `/applet/${appletName}`;

  const props = {
    apiPath,
  }

  import(`./${component}.vue`).then((module) => {
    appletProps.value = props;
    applet.value = defineAsyncComponent(() => Promise.resolve(module.default));
  });
}

const close = () => {
  console.log("close");
  backend.post(`${appletProps.value.apiPath}/close`, {});
};

const clearApplet = () => {
  applet.value = null;
};

const openWebsocket = () => {
  const seq = ++websocketSeq;
  console.log("Connecting to websocket @ ", webSocketUrl);
  if (websocket) {
    websocket.close();
    websocket = null;
  }
  websocket = new WebSocket(webSocketUrl);
  websocket.onmessage = (event) => {
    if (seq != websocketSeq) return;
    console.log('Websocket data', event.data);
    const message = JSON.parse(event.data);
    console.log('Websocket message', message);
    if (message.command == 'activate') {
      activate(message.applet, message.path, message.component);
    } else if (message.command == 'deactivate') {
      clearApplet();
    } else if (message.command == 'log') {
      log.value.push(message);
    } else if (message.command == 'connect_log') {
      logger_name.value = message.name;
      log.value.length = 0;
    } else if (message.command == 'disconnect_log') {
      logger_name.value = null;
      log.value.length = 0;
    }
  };
  websocket.onopen = () => {
    if (seq != websocketSeq) return;
    //console.log('Websocket opened');
    isConnected.value = true;
  };
  websocket.onclose = () => {
    if (seq != websocketSeq) return;
    //console.log('Websocket closed');
    isConnected.value = false;
    clearApplet()
  };
  websocket.onerror = (error) => {
    if (seq != websocketSeq) return;
    //console.error('Websocket error:', error);
  };
};

const closeWebsocket = () => {
  websocket.close();
  websocket = null;
};

setInterval(() => {
  if (shouldBeConnected && !isConnected.value) {
    openWebsocket();
  }
}, 2000);

onMounted(() => {
  shouldBeConnected = true;
  openWebsocket();
});

onBeforeUnmount(() => {
  closeWebsocket();
  shouldBeConnected = false;
});
</script>
