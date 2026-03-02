"use client"

import React, { useState, useRef, useEffect, useCallback } from 'react'
import { Mic, Square, Upload, Loader2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

interface VoiceRecorderProps {
  onRecordingComplete: (blob: Blob) => void
  maxDuration?: number
  className?: string
}

export function VoiceRecorder({
  onRecordingComplete,
  maxDuration = 30,
  className,
}: VoiceRecorderProps) {
  const [isRecording, setIsRecording] = useState(false)
  const [recordingTime, setRecordingTime] = useState(0)
  const [audioLevel, setAudioLevel] = useState(0)
  const [permissionDenied, setPermissionDenied] = useState(false)
  
  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const streamRef = useRef<MediaStream | null>(null)
  const chunksRef = useRef<Blob[]>([])
  const timerRef = useRef<NodeJS.Timeout | null>(null)
  const analyserRef = useRef<AnalyserNode | null>(null)
  const animationRef = useRef<number | null>(null)

  const updateAudioLevel = useCallback(() => {
    if (!analyserRef.current) return
    
    const dataArray = new Uint8Array(analyserRef.current.frequencyBinCount)
    analyserRef.current.getByteFrequencyData(dataArray)
    
    const average = dataArray.reduce((a, b) => a + b, 0) / dataArray.length
    setAudioLevel(average / 255)
    
    animationRef.current = requestAnimationFrame(updateAudioLevel)
  }, [])

  const startRecording = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
      streamRef.current = stream
      
      const audioContext = new AudioContext()
      const source = audioContext.createMediaStreamSource(stream)
      const analyser = audioContext.createAnalyser()
      analyser.fftSize = 256
      source.connect(analyser)
      analyserRef.current = analyser
      
      const mediaRecorder = new MediaRecorder(stream, {
        mimeType: 'audio/webm;codecs=opus',
      })
      mediaRecorderRef.current = mediaRecorder
      
      chunksRef.current = []
      
      mediaRecorder.ondataavailable = (e) => {
        if (e.data.size > 0) {
          chunksRef.current.push(e.data)
        }
      }
      
      mediaRecorder.onstop = () => {
        const blob = new Blob(chunksRef.current, { type: 'audio/webm' })
        onRecordingComplete(blob)
        
        stream.getTracks().forEach(track => track.stop())
        if (animationRef.current) {
          cancelAnimationFrame(animationRef.current)
        }
      }
      
      mediaRecorder.start(100)
      setIsRecording(true)
      setRecordingTime(0)
      
      updateAudioLevel()
      
      timerRef.current = setInterval(() => {
        setRecordingTime(prev => {
          if (prev >= maxDuration) {
            stopRecording()
            return prev
          }
          return prev + 1
        })
      }, 1000)
      
    } catch (err) {
      console.error('Error accessing microphone:', err)
      setPermissionDenied(true)
    }
  }

  const stopRecording = () => {
    if (mediaRecorderRef.current && isRecording) {
      mediaRecorderRef.current.stop()
      setIsRecording(false)
      setAudioLevel(0)
      
      if (timerRef.current) {
        clearInterval(timerRef.current)
      }
    }
  }

  useEffect(() => {
    return () => {
      if (timerRef.current) {
        clearInterval(timerRef.current)
      }
      if (animationRef.current) {
        cancelAnimationFrame(animationRef.current)
      }
      if (streamRef.current) {
        streamRef.current.getTracks().forEach(track => track.stop())
      }
    }
  }, [])

  const formatTime = (seconds: number) => {
    const mins = Math.floor(seconds / 60)
    const secs = seconds % 60
    return `${mins}:${secs.toString().padStart(2, '0')}`
  }

  return (
    <div className={cn("space-y-4", className)}>
      <div className="flex items-center justify-center gap-4">
        {!isRecording ? (
          <Button
            variant="cyber"
            size="lg"
            onClick={startRecording}
            disabled={permissionDenied}
            className="gap-2"
          >
            <Mic className="h-5 w-5" />
            Start Recording
          </Button>
        ) : (
          <Button
            variant="destructive"
            size="lg"
            onClick={stopRecording}
            className="gap-2"
          >
            <Square className="h-5 w-5" />
            Stop
          </Button>
        )}
      </div>

      {permissionDenied && (
        <p className="text-sm text-destructive text-center">
          Microphone access denied. Please enable it in your browser settings.
        </p>
      )}

      {isRecording && (
        <div className="space-y-3">
          <div className="text-center font-mono text-cyber-cyan">
            {formatTime(recordingTime)} / {formatTime(maxDuration)}
          </div>
          
          <div className="flex items-center justify-center gap-1 h-16">
            {Array.from({ length: 20 }).map((_, i) => (
              <div
                key={i}
                className="w-1.5 bg-gradient-to-t from-cyber-cyan to-cyber-purple rounded-full transition-all duration-75"
                style={{
                  height: `${Math.max(4, Math.random() * audioLevel * 100)}%`,
                  opacity: 0.3 + audioLevel * 0.7,
                }}
              />
            ))}
          </div>
          
          <Progress percentage={(recordingTime / maxDuration) * 100} />
        </div>
      )}
    </div>
  )
}

function Progress({ percentage }: { percentage: number }) {
  return (
    <div className="h-1 w-full bg-muted rounded-full overflow-hidden">
      <div
        className="h-full bg-gradient-to-r from-cyber-cyan to-cyber-purple transition-all duration-300"
        style={{ width: `${percentage}%` }}
      />
    </div>
  )
}
