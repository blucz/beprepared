<template>
  <div id="app" class="bg-dark text-white d-flex flex-column vh-100">
      <!-- Header -->
    <header class="header p-2">
      <img src='/beprepared.jpg' alt='beprepared' height='60'/>
    </header>

    <!-- Main content area -->
    <main class="flex-grow-1 d-flex flex-column bg-secondary p-3">
      <div v-if='!isConnected' class="text-center my-4">
        <h2>beprepared!</h2>
      </div>

      <div v-if='isConnected && !applet' class="container-fluid px-3">
        <div v-if='!logger_name' class="text-center my-4">
          <div class="spinner-border" role="status">
            <span class="visually-hidden">Loading...</span>
          </div>
        </div>
        <div v-else class="terminal-wrapper">
          <h2 class="text-center mb-2">{{ logger_name }}</h2>
          <div v-if="progress" class="progress-info mb-4">
              <div class="progress-header">
                <span class="progress-desc">{{ progress.desc }}</span>
                <span class="progress-stats">
                  {{ progress.n }}{{ progress.total ? '/' + progress.total : '' }}
                  <span v-if="progress.rate" class="progress-rate">
                    &#8227; {{ progress.rate >= 1 ? `${progress.rate.toFixed(2)} it/s` : `${(1/progress.rate).toFixed(2)}s/it` }}
                  </span>
                  <span v-if="progress.total && progress.rate" class="progress-time">
                    &#8227; ETA: {{ formatTime((progress.total - progress.n) / progress.rate) }}
                  </span>
                </span>
              </div>
              <div class="progress" style="height: 8px;">
                <div class="progress-bar" role="progressbar" 
                  :class="{ 'progress-bar-striped progress-bar-animated': !progress.total }"
                  :style="{ width: progressPercent + '%' }" 
                  :aria-valuenow="progress.n"
                  aria-valuemin="0" 
                  :aria-valuemax="progress.total || 100">
                </div>
              </div>
            </div>
          <div class="terminal-container" v-if="log.length > 0" @scroll="handleScroll">
            <div class="terminal bg-black rounded p-2">
              <div v-for='entry in log' :key='entry.id' class="log-entry">
                <span class="timestamp">[{{ entry.time }}]</span> {{ entry.message }}
              </div>
            </div>
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
.terminal-wrapper {
  width: 100%;
  max-width: 1200px;
  margin: 0 auto;
}
.terminal-container {
  width: 100%;
  border: 1px solid #30363d;
  border-radius: 6px;
  background: #0d1117;
  box-shadow: 0 0 10px rgba(0,0,0,0.3);
  max-height: calc(1.3em * 40); /* 40 rows of text */
  overflow-y: auto;
  overflow-x: hidden;
}
.progress-info {
  width: 100%;
  margin: 8px 0;
  padding: 12px;
  background: #0d1117;
  border: 1px solid #30363d;
  border-radius: 6px;
}
.progress-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  font-size: 0.9em;
}
.progress-desc {
  color: #e6e6e6;
}
.progress-stats {
  color: #8b949e;
}
.progress-rate, .progress-time {
  margin-left: 10px;
  color: #6e7681;
}
.progress {
  background-color: #21262d;
  border: 1px solid #30363d;
  border-radius: 3px;
  overflow: hidden;
}
.progress-bar {
  background-color: #238636;
  transition: width 0.3s ease;
}
.terminal {
  height: 100%;
  width: 100%;
  overflow-y: auto;
  overflow-x: hidden;
  font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
  font-size: 0.9em;
  line-height: 1.3;
  color: #e6e6e6;
}
.log-entry {
  padding: 1px 0;
  white-space: pre-wrap;
  word-wrap: break-word;
  max-width: 100%;
  overflow-wrap: break-word;
}
.timestamp {
  color: #6e7681;
  user-select: none;
}
/* Custom scrollbar for the terminal */
.terminal::-webkit-scrollbar {
  width: 8px;
}
.terminal::-webkit-scrollbar-track {
  background: #161b22;
}
.terminal::-webkit-scrollbar-thumb {
  background: #30363d;
  border-radius: 4px;
}
.terminal::-webkit-scrollbar-thumb:hover {
  background: #3f444d;
}
</style>

<script setup>
import { ref, shallowRef, computed, onMounted, onBeforeUnmount, createVNode, render, nextTick } from 'vue';
import { createApp, defineAsyncComponent, markRaw } from 'vue';
import axios from 'axios';

// Import all known applet components
import HumanTag from './HumanTag.vue';
import HumanFilter from './HumanFilter.vue';
import Generic from './Generic.vue';


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
const userHasScrolled = ref(false);
const progress = ref(null);

const formatTime = (seconds) => {
  if (!seconds || isNaN(seconds)) return '??:??';
  seconds = Math.floor(seconds);
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = seconds % 60;
  if (hours > 0) {
    return `${String(hours).padStart(2, '0')}:${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
  }
  return `${String(minutes).padStart(2, '0')}:${String(secs).padStart(2, '0')}`;
};

const progressPercent = computed(() => {
  if (!progress.value) return 0;
  if (!progress.value.total) {
    // Show indeterminate progress when no total
    return 100;
  }
  return (progress.value.n / progress.value.total) * 100;
});

let shouldBeConnected = false;
let websocket;
let websocketSeq = 0;

const activate = async (appletName, path, component) => {
  console.log(`Activating applet ${appletName} ${path} ${component}`);
  const apiPath = `/applet/${appletName}`;

  const props = {
    apiPath,
  }

  // Switch based on component name
  switch(component) {
    case 'HumanTag':
      applet.value = HumanTag;
      break;
    case 'HumanFilter':
      applet.value = HumanFilter;
      break;
    default:
      applet.value = Generic;
      break;
  }
  appletProps.value = props;
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
  
  // Set a timeout for the connection
  const connectionTimeout = setTimeout(() => {
    if (websocket && websocket.readyState === WebSocket.CONNECTING) {
      console.log("WebSocket connection timeout");
      websocket.close();
    }
  }, 5000); // 5 second timeout

  websocket.onopen = () => {
    clearTimeout(connectionTimeout);
    if (seq != websocketSeq) return;
    console.log('WebSocket connection established');
    isConnected.value = true;
    // Clear any pending reconnection attempts
    websocketSeq = seq;
  };

  websocket.onmessage = (event) => {
    if (seq != websocketSeq) return;
    console.log('Websocket data', event.data);
    const message = JSON.parse(event.data);
    console.log('Websocket message', message);
    if (message.command == 'history') {
      // Restore history
      log.value = message.logs;
      progress.value = message.progress;
    } else if (message.command == 'activate') {
      activate(message.applet, message.path, message.component);
    } else if (message.command == 'deactivate') {
      clearApplet();
    } else if (message.command == 'log') {
      log.value.push({
        ...message,
        time: new Date().toLocaleTimeString()
      });
      // Auto-scroll to bottom if user hasn't scrolled up
      nextTick(() => {
        const terminal = document.querySelector('.terminal-container');
        if (terminal) {
          const isScrolledToBottom = Math.abs(terminal.scrollHeight - terminal.clientHeight - terminal.scrollTop) < 1;
          if (!userHasScrolled.value || isScrolledToBottom) {
            terminal.scrollTop = terminal.scrollHeight;
          }
        }
      });
    } else if (message.command == 'connect_log') {
      logger_name.value = message.name;
      log.value.length = 0;
    } else if (message.command == 'disconnect_log') {
      logger_name.value = null;
      log.value.length = 0;
    } else if (message.command == 'progress') {
      if (message.clear) {
        progress.value = null;
      } else {
        progress.value = message;
      }
    }
  };
  websocket.onclose = (event) => {
    clearTimeout(connectionTimeout);
    if (seq != websocketSeq) return;
    console.log('WebSocket closed', event.code, event.reason);
    isConnected.value = false;
    clearApplet();
    
    // Only reconnect if this was the most recent connection attempt
    if (shouldBeConnected && seq === websocketSeq) {
      setTimeout(() => {
        if (shouldBeConnected && seq === websocketSeq) {
          console.log('Attempting to reconnect WebSocket...');
          openWebsocket();
        }
      }, 1000); // 1 second delay before reconnecting
    }
  };

  websocket.onerror = (error) => {
    clearTimeout(connectionTimeout);
    if (seq != websocketSeq) return;
    console.error('WebSocket error:', error);
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

const handleScroll = (event) => {
  const terminal = event.target;
  const isScrolledToBottom = Math.abs(terminal.scrollHeight - terminal.clientHeight - terminal.scrollTop) < 1;
  if (!isScrolledToBottom) {
    userHasScrolled.value = true;
  } else {
    userHasScrolled.value = false;
  }
};

onBeforeUnmount(() => {
  shouldBeConnected = false;
  closeWebsocket();
});
</script>
