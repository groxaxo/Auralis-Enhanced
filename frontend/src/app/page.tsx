"use client"

import React, { useState, useCallback } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  Sparkles, 
  Wand2, 
  ChevronDown, 
  ChevronUp,
  Download,
  Loader2,
  AlertCircle,
  CheckCircle2
} from 'lucide-react'
import { Textarea } from '@/components/ui/textarea'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { 
  Select, 
  SelectContent, 
  SelectItem, 
  SelectTrigger, 
  SelectValue 
} from '@/components/ui/select'
import { ParameterSlider, ParameterToggle } from '@/components/audio/parameter-controls'
import { AudioUpload } from '@/components/audio/audio-upload'
import { WaveformPlayer } from '@/components/audio/waveform-player'
import { VoiceRecorder } from '@/components/audio/voice-recorder'
import { Progress } from '@/components/ui/progress'
import { auralisAPI } from '@/lib/api'
import { useGenerationStore, useAppStore } from '@/stores'
import { SUPPORTED_LANGUAGES } from '@/types'
import { cn } from '@/lib/utils'

export default function GeneratePage() {
  const [text, setText] = useState('')
  const [voiceFiles, setVoiceFiles] = useState<File[]>([])
  const [showAdvanced, setShowAdvanced] = useState(false)
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null)
  
  const [settings, setSettings] = useState({
    speed: 1.0,
    enhance_speech: false,
    apply_novasr: false,
    temperature: 0.75,
    top_p: 0.85,
    top_k: 50,
    repetition_penalty: 5.0,
    language: 'auto',
  })
  
  const { isGenerating, progress, stage, error, audioUrl, setGenerating, setProgress, setError, setAudioUrl, reset } = useGenerationStore()
  const { apiBaseUrl } = useAppStore()

  const handleGenerate = async () => {
    if (!text.trim()) {
      setError('Please enter some text to generate')
      return
    }
    
    if (voiceFiles.length === 0 && !recordedBlob) {
      setError('Please provide at least one reference audio file or record your voice')
      return
    }

    reset()
    setGenerating(true)
    setError(null)

    try {
      console.log('[GeneratePage] Starting generation...')
      console.log('[GeneratePage] API Base URL:', apiBaseUrl)
      
      auralisAPI.setBaseUrl(apiBaseUrl)
      
      const speakerFiles: File[] = recordedBlob 
        ? [new File([recordedBlob], 'recording.webm', { type: 'audio/webm' })]
        : voiceFiles
      
      console.log('[GeneratePage] Speaker files:', speakerFiles.map(f => ({
        name: f.name,
        type: f.type,
        size: f.size,
      })))
      
      const request = {
        text,
        speaker_files: speakerFiles,
        ...settings,
      }
      
      console.log('[GeneratePage] Full request:', {
        ...request,
        speaker_files: `${speakerFiles.length} files`,
      })

      setProgress(10, 'preparing')
      
      const response = await auralisAPI.generateSpeech(request)
      
      console.log('[GeneratePage] Response received:', response)
      
      setProgress(100, 'complete')
      setAudioUrl(response.audioUrl)
      
    } catch (err) {
      console.error('[GeneratePage] Generation error:', err)
      const errorMessage = err instanceof Error ? err.message : 'Generation failed'
      setError(errorMessage)
      setGenerating(false)
    }
  }

  const handleDownload = () => {
    if (!audioUrl) return
    
    const a = document.createElement('a')
    a.href = audioUrl
    a.download = `auralis-output-${Date.now()}.wav`
    a.click()
  }

  const getStageMessage = () => {
    switch (stage) {
      case 'preparing': return 'Preparing audio generation...'
      case 'processing': return 'Processing text and generating speech...'
      case 'enhancing': return 'Applying NovaSR enhancement (48kHz)...'
      case 'complete': return 'Generation complete!'
      case 'error': return error || 'An error occurred'
      default: return 'Ready'
    }
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex items-center gap-3 mb-2">
          <Sparkles className="h-8 w-8 text-cyber-cyan" />
          <h1 className="text-3xl font-bold text-gradient-cyber">Generate Speech</h1>
        </div>
        <p className="text-muted-foreground">
          Transform your text into natural speech with voice cloning
        </p>
      </motion.div>

      <div className="grid lg:grid-cols-2 gap-6">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
          className="space-y-6"
        >
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Text Input</CardTitle>
              <CardDescription>
                Enter the text you want to convert to speech
              </CardDescription>
            </CardHeader>
            <CardContent>
              <Textarea
                placeholder="Enter your text here..."
                value={text}
                onChange={(e) => setText(e.target.value)}
                className="min-h-[200px] resize-y"
              />
              <div className="flex justify-between mt-2 text-xs text-muted-foreground">
                <span>{text.length} characters</span>
                <span>~{Math.ceil(text.split(/\s+/).filter(Boolean).length / 150)} min audio</span>
              </div>
            </CardContent>
          </Card>

          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Reference Voice</CardTitle>
              <CardDescription>
                Upload audio files or record your voice for cloning
              </CardDescription>
            </CardHeader>
            <CardContent className="space-y-4">
              <AudioUpload
                files={voiceFiles}
                onFilesSelected={setVoiceFiles}
              />
              
              <div className="relative">
                <div className="absolute inset-0 flex items-center">
                  <div className="w-full border-t border-border" />
                </div>
                <div className="relative flex justify-center text-xs uppercase">
                  <span className="bg-card px-2 text-muted-foreground">Or record</span>
                </div>
              </div>
              
              <VoiceRecorder
                onRecordingComplete={(blob) => {
                  setRecordedBlob(blob)
                  setVoiceFiles([])
                }}
              />
              
              {recordedBlob && (
                <div className="flex items-center gap-2 p-2 rounded bg-accent/10 text-sm">
                  <CheckCircle2 className="h-4 w-4 text-green-500" />
                  <span>Recording ready ({(recordedBlob.size / 1024).toFixed(1)} KB)</span>
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="space-y-6"
        >
          <Card>
            <CardHeader>
              <CardTitle className="text-lg">Basic Settings</CardTitle>
            </CardHeader>
            <CardContent className="space-y-6">
              <div className="space-y-2">
                <label className="text-sm font-medium">Language</label>
                <Select
                  value={settings.language}
                  onValueChange={(v) => setSettings(s => ({ ...s, language: v }))}
                >
                  <SelectTrigger>
                    <SelectValue />
                  </SelectTrigger>
                  <SelectContent>
                    {SUPPORTED_LANGUAGES.map(lang => (
                      <SelectItem key={lang.code} value={lang.code}>
                        {lang.name}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>

              <ParameterSlider
                label="Speed"
                value={settings.speed}
                onChange={(v) => setSettings(s => ({ ...s, speed: v }))}
                min={0.5}
                max={2.0}
                step={0.1}
                unit="x"
              />

              <ParameterToggle
                label="NovaSR Enhancement"
                checked={settings.apply_novasr}
                onChange={(v) => setSettings(s => ({ ...s, apply_novasr: v }))}
                description="Upscale to 48kHz professional quality"
                badge="48kHz"
              />

              <ParameterToggle
                label="Enhance Reference Speech"
                checked={settings.enhance_speech}
                onChange={(v) => setSettings(s => ({ ...s, enhance_speech: v }))}
                description="Improve quality of reference audio"
              />
            </CardContent>
          </Card>

          <Card>
            <button
              onClick={() => setShowAdvanced(!showAdvanced)}
              className="w-full p-6 flex items-center justify-between hover:bg-accent/5 transition-colors rounded-lg"
            >
              <CardTitle className="text-lg">Advanced Settings</CardTitle>
              {showAdvanced ? (
                <ChevronUp className="h-5 w-5 text-muted-foreground" />
              ) : (
                <ChevronDown className="h-5 w-5 text-muted-foreground" />
              )}
            </button>
            
            <AnimatePresence>
              {showAdvanced && (
                <motion.div
                  initial={{ height: 0, opacity: 0 }}
                  animate={{ height: 'auto', opacity: 1 }}
                  exit={{ height: 0, opacity: 0 }}
                  transition={{ duration: 0.3 }}
                  className="overflow-hidden"
                >
                  <CardContent className="space-y-6 pt-0">
                    <ParameterSlider
                      label="Temperature"
                      value={settings.temperature}
                      onChange={(v) => setSettings(s => ({ ...s, temperature: v }))}
                      min={0.5}
                      max={1.0}
                      step={0.05}
                      description="Controls randomness in generation"
                    />

                    <ParameterSlider
                      label="Top P"
                      value={settings.top_p}
                      onChange={(v) => setSettings(s => ({ ...s, top_p: v }))}
                      min={0.5}
                      max={1.0}
                      step={0.05}
                      description="Nucleus sampling threshold"
                    />

                    <ParameterSlider
                      label="Top K"
                      value={settings.top_k}
                      onChange={(v) => setSettings(s => ({ ...s, top_k: v }))}
                      min={0}
                      max={100}
                      step={5}
                      description="Top-k sampling parameter"
                    />

                    <ParameterSlider
                      label="Repetition Penalty"
                      value={settings.repetition_penalty}
                      onChange={(v) => setSettings(s => ({ ...s, repetition_penalty: v }))}
                      min={1.0}
                      max={10.0}
                      step={0.5}
                      description="Prevents repetitive output"
                    />
                  </CardContent>
                </motion.div>
              )}
            </AnimatePresence>
          </Card>

          <Button
            variant="cyber"
            size="lg"
            className="w-full gap-2"
            onClick={handleGenerate}
            disabled={isGenerating || !text.trim()}
          >
            {isGenerating ? (
              <>
                <Loader2 className="h-5 w-5 animate-spin" />
                Generating...
              </>
            ) : (
              <>
                <Wand2 className="h-5 w-5" />
                Generate Speech
              </>
            )}
          </Button>

          {isGenerating && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="space-y-2"
            >
              <div className="flex items-center justify-between text-sm">
                <span className="text-muted-foreground">{getStageMessage()}</span>
                <span className="font-mono text-cyber-cyan">{progress}%</span>
              </div>
              <Progress value={progress} />
            </motion.div>
          )}

          {error && (
            <motion.div
              initial={{ opacity: 0, y: 10 }}
              animate={{ opacity: 1, y: 0 }}
              className="flex items-center gap-2 p-4 rounded-lg bg-destructive/10 border border-destructive/20 text-destructive"
            >
              <AlertCircle className="h-5 w-5 flex-shrink-0" />
              <span className="text-sm">{error}</span>
            </motion.div>
          )}
        </motion.div>
      </div>

      {audioUrl && (
        <motion.div
          initial={{ opacity: 0, y: 20 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.5 }}
        >
          <Card className="border-cyber-cyan/30">
            <CardHeader className="flex flex-row items-center justify-between">
              <div>
                <CardTitle className="text-lg">Generated Audio</CardTitle>
                <CardDescription>
                  {settings.apply_novasr ? '48kHz NovaSR Enhanced' : '24kHz Standard'}
                </CardDescription>
              </div>
              <Button variant="outline" size="sm" onClick={handleDownload} className="gap-2">
                <Download className="h-4 w-4" />
                Download
              </Button>
            </CardHeader>
            <CardContent>
              <WaveformPlayer audioUrl={audioUrl} />
            </CardContent>
          </Card>
        </motion.div>
      )}
    </div>
  )
}
