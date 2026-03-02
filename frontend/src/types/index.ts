export interface Voice {
  id: string
  name: string
  audioPath: string
  previewUrl?: string
  createdAt: Date
  metadata?: {
    duration?: number
    sampleRate?: number
    language?: string
  }
}

export interface TTSRequest {
  text: string
  speaker_files: File[]
  language: string
  speed: number
  enhance_speech: boolean
  apply_novasr: boolean
  temperature: number
  top_p: number
  top_k: number
  repetition_penalty: number
  stream?: boolean
}

export interface TTSResponse {
  audioUrl: string
  sampleRate: number
  duration: number
  novasr_applied: boolean
}

export interface GenerationProgress {
  stage: 'preparing' | 'processing' | 'enhancing' | 'complete' | 'error'
  progress: number
  message: string
  audioChunk?: ArrayBuffer
}

export interface TTSPreset {
  id: string
  name: string
  settings: Omit<TTSRequest, 'text' | 'speaker_files'>
  voiceId?: string
}

export interface APIConfig {
  baseUrl: string
  timeout: number
}

export const SUPPORTED_LANGUAGES = [
  { code: 'auto', name: 'Auto Detect' },
  { code: 'en', name: 'English' },
  { code: 'es', name: 'Spanish' },
  { code: 'fr', name: 'French' },
  { code: 'de', name: 'German' },
  { code: 'it', name: 'Italian' },
  { code: 'pt', name: 'Portuguese' },
  { code: 'pl', name: 'Polish' },
  { code: 'tr', name: 'Turkish' },
  { code: 'ru', name: 'Russian' },
  { code: 'nl', name: 'Dutch' },
  { code: 'cs', name: 'Czech' },
  { code: 'ar', name: 'Arabic' },
  { code: 'zh-cn', name: 'Chinese (Simplified)' },
  { code: 'hu', name: 'Hungarian' },
  { code: 'ko', name: 'Korean' },
  { code: 'ja', name: 'Japanese' },
  { code: 'hi', name: 'Hindi' },
] as const

export const DEFAULT_SETTINGS = {
  speed: 1.0,
  enhance_speech: false,
  apply_novasr: false,
  temperature: 0.75,
  top_p: 0.85,
  top_k: 50,
  repetition_penalty: 5.0,
  language: 'auto',
} as const
