"use client"

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Mic, Upload, Save, Play, Trash2, CheckCircle2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { VoiceRecorder } from '@/components/audio/voice-recorder'
import { AudioUpload } from '@/components/audio/audio-upload'
import { WaveformPlayer } from '@/components/audio/waveform-player'
import { useAppStore } from '@/stores'
import { v4 as uuidv4 } from 'uuid'
import type { Voice } from '@/types'

export default function VoiceCloningPage() {
  const [recordedBlob, setRecordedBlob] = useState<Blob | null>(null)
  const [uploadedFiles, setUploadedFiles] = useState<File[]>([])
  const [voiceName, setVoiceName] = useState('')
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [saved, setSaved] = useState(false)
  
  const { addVoice } = useAppStore()

  const handleSaveVoice = () => {
    if (!voiceName.trim()) return
    if (!recordedBlob && uploadedFiles.length === 0) return

    const voice: Voice = {
      id: uuidv4(),
      name: voiceName,
      audioPath: recordedBlob 
        ? URL.createObjectURL(recordedBlob) 
        : URL.createObjectURL(uploadedFiles[0]),
      previewUrl: recordedBlob 
        ? URL.createObjectURL(recordedBlob) 
        : URL.createObjectURL(uploadedFiles[0]),
      createdAt: new Date(),
    }

    addVoice(voice)
    setSaved(true)
    
    setTimeout(() => {
      setSaved(false)
      setRecordedBlob(null)
      setUploadedFiles([])
      setVoiceName('')
      setPreviewUrl(null)
    }, 2000)
  }

  const handlePreview = () => {
    if (recordedBlob) {
      setPreviewUrl(URL.createObjectURL(recordedBlob))
    } else if (uploadedFiles.length > 0) {
      setPreviewUrl(URL.createObjectURL(uploadedFiles[0]))
    }
  }

  const canSave = voiceName.trim() && (recordedBlob || uploadedFiles.length > 0)

  return (
    <div className="max-w-4xl mx-auto space-y-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex items-center gap-3 mb-2">
          <Mic className="h-8 w-8 text-cyber-purple" />
          <h1 className="text-3xl font-bold text-gradient-cyber">Voice Cloning</h1>
        </div>
        <p className="text-muted-foreground">
          Create a custom voice by recording or uploading audio samples
        </p>
      </motion.div>

      <div className="grid md:grid-cols-2 gap-6">
        <motion.div
          initial={{ opacity: 0, x: -20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.1 }}
        >
          <Card className="h-full">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Mic className="h-5 w-5 text-cyber-cyan" />
                Record Voice
              </CardTitle>
              <CardDescription>
                Record a 10-30 second sample of your voice for best results
              </CardDescription>
            </CardHeader>
            <CardContent>
              <VoiceRecorder
                onRecordingComplete={(blob) => {
                  setRecordedBlob(blob)
                  setUploadedFiles([])
                  setPreviewUrl(null)
                }}
                maxDuration={30}
              />
              {recordedBlob && (
                <div className="mt-4 flex items-center gap-2 text-sm text-green-500">
                  <CheckCircle2 className="h-4 w-4" />
                  Recording complete - {(recordedBlob.size / 1024).toFixed(1)} KB
                </div>
              )}
            </CardContent>
          </Card>
        </motion.div>

        <motion.div
          initial={{ opacity: 0, x: 20 }}
          animate={{ opacity: 1, x: 0 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <Card className="h-full">
            <CardHeader>
              <CardTitle className="text-lg flex items-center gap-2">
                <Upload className="h-5 w-5 text-cyber-purple" />
                Upload Audio
              </CardTitle>
              <CardDescription>
                Upload existing audio files for voice cloning
              </CardDescription>
            </CardHeader>
            <CardContent>
              <AudioUpload
                files={uploadedFiles}
                onFilesSelected={(files) => {
                  setUploadedFiles(files)
                  setRecordedBlob(null)
                  setPreviewUrl(null)
                }}
                maxFiles={5}
              />
            </CardContent>
          </Card>
        </motion.div>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.3 }}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Save Voice Profile</CardTitle>
            <CardDescription>
              Give your voice profile a name and save it to your library
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="voice-name">Voice Name</Label>
              <Input
                id="voice-name"
                placeholder="e.g., My Custom Voice"
                value={voiceName}
                onChange={(e) => setVoiceName(e.target.value)}
              />
            </div>

            {(recordedBlob || uploadedFiles.length > 0) && (
              <Button
                variant="outline"
                onClick={handlePreview}
                className="gap-2"
              >
                <Play className="h-4 w-4" />
                Preview Audio
              </Button>
            )}

            {previewUrl && (
              <WaveformPlayer audioUrl={previewUrl} height={60} />
            )}

            <div className="flex gap-3">
              <Button
                variant="cyber"
                onClick={handleSaveVoice}
                disabled={!canSave || saved}
                className="gap-2"
              >
                {saved ? (
                  <>
                    <CheckCircle2 className="h-4 w-4" />
                    Saved!
                  </>
                ) : (
                  <>
                    <Save className="h-4 w-4" />
                    Save Voice Profile
                  </>
                )}
              </Button>
              
              {(recordedBlob || uploadedFiles.length > 0) && (
                <Button
                  variant="outline"
                  onClick={() => {
                    setRecordedBlob(null)
                    setUploadedFiles([])
                    setPreviewUrl(null)
                  }}
                  className="gap-2"
                >
                  <Trash2 className="h-4 w-4" />
                  Clear
                </Button>
              )}
            </div>
          </CardContent>
        </Card>
      </motion.div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.4 }}
        className="p-4 rounded-lg bg-gradient-to-r from-cyber-cyan/10 to-cyber-purple/10 border border-cyber-cyan/20"
      >
        <h3 className="font-semibold mb-2">Tips for Best Results</h3>
        <ul className="text-sm text-muted-foreground space-y-1">
          <li>• Record in a quiet environment with minimal background noise</li>
          <li>• Speak naturally and clearly for 10-30 seconds</li>
          <li>• Include varied intonation and pacing</li>
          <li>• Higher quality input audio produces better cloning results</li>
        </ul>
      </motion.div>
    </div>
  )
}
