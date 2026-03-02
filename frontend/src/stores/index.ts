import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import type { Voice, TTSPreset, TTSRequest, DEFAULT_SETTINGS } from '@/types'

const getDefaultApiUrl = () => {
  if (typeof window !== 'undefined') {
    return `http://${window.location.hostname}:8001`
  }
  return 'http://localhost:8001'
}

interface AppState {
  voices: Voice[]
  presets: TTSPreset[]
  defaultSettings: typeof DEFAULT_SETTINGS
  apiBaseUrl: string
  theme: 'dark' | 'light'
  
  addVoice: (voice: Voice) => void
  removeVoice: (id: string) => void
  updateVoice: (id: string, updates: Partial<Voice>) => void
  
  addPreset: (preset: TTSPreset) => void
  removePreset: (id: string) => void
  updatePreset: (id: string, updates: Partial<TTSPreset>) => void
  
  setApiBaseUrl: (url: string) => void
  setTheme: (theme: 'dark' | 'light') => void
}

export const useAppStore = create<AppState>()(
  persist(
    (set) => ({
      voices: [],
      presets: [],
      defaultSettings: {
        speed: 1.0,
        enhance_speech: false,
        apply_novasr: false,
        temperature: 0.75,
        top_p: 0.85,
        top_k: 50,
        repetition_penalty: 5.0,
        language: 'auto',
      },
      apiBaseUrl: getDefaultApiUrl(),
      theme: 'dark',
      
      addVoice: (voice) => set((state) => ({ 
        voices: [...state.voices, voice] 
      })),
      
      removeVoice: (id) => set((state) => ({ 
        voices: state.voices.filter((v) => v.id !== id) 
      })),
      
      updateVoice: (id, updates) => set((state) => ({ 
        voices: state.voices.map((v) => 
          v.id === id ? { ...v, ...updates } : v
        ) 
      })),
      
      addPreset: (preset) => set((state) => ({ 
        presets: [...state.presets, preset] 
      })),
      
      removePreset: (id) => set((state) => ({ 
        presets: state.presets.filter((p) => p.id !== id) 
      })),
      
      updatePreset: (id, updates) => set((state) => ({ 
        presets: state.presets.map((p) => 
          p.id === id ? { ...p, ...updates } : p
        ) 
      })),
      
      setApiBaseUrl: (url) => set({ apiBaseUrl: url }),
      setTheme: (theme) => set({ theme }),
    }),
    {
      name: 'auralis-storage',
      partialize: (state) => ({
        voices: state.voices,
        presets: state.presets,
        apiBaseUrl: state.apiBaseUrl,
        theme: state.theme,
      }),
      merge: (persisted, current) => {
        const merged = { ...current, ...(persisted || {}) } as AppState
        if (typeof window !== 'undefined' && merged.apiBaseUrl) {
          const hostname = window.location.hostname
          if (!merged.apiBaseUrl.includes(hostname) && merged.apiBaseUrl.includes('localhost')) {
            merged.apiBaseUrl = `http://${hostname}:8001`
          }
        }
        return merged
      },
    }
  )
)

interface GenerationState {
  isGenerating: boolean
  progress: number
  stage: string
  error: string | null
  audioUrl: string | null
  
  setGenerating: (isGenerating: boolean) => void
  setProgress: (progress: number, stage: string) => void
  setError: (error: string | null) => void
  setAudioUrl: (url: string | null) => void
  reset: () => void
}

export const useGenerationStore = create<GenerationState>((set) => ({
  isGenerating: false,
  progress: 0,
  stage: 'idle',
  error: null,
  audioUrl: null,
  
  setGenerating: (isGenerating) => set({ isGenerating }),
  setProgress: (progress, stage) => set({ progress, stage }),
  setError: (error) => set({ error, isGenerating: false }),
  setAudioUrl: (url) => set({ audioUrl: url, isGenerating: false, progress: 100, stage: 'complete' }),
  reset: () => set({ 
    isGenerating: false, 
    progress: 0, 
    stage: 'idle', 
    error: null, 
    audioUrl: null 
  }),
}))
