"use client"

import React, { useEffect, useRef, useCallback, useState } from 'react'
import WaveSurfer from 'wavesurfer.js'
import { Play, Pause, SkipBack, Volume2, VolumeX } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Slider } from '@/components/ui/slider'
import { cn } from '@/lib/utils'

interface WaveformPlayerProps {
  audioUrl?: string | null
  audioBlob?: Blob | null
  height?: number
  className?: string
  onReady?: () => void
  onFinish?: () => void
}

export function WaveformPlayer({
  audioUrl,
  audioBlob,
  height = 80,
  className,
  onReady,
  onFinish,
}: WaveformPlayerProps) {
  const containerRef = useRef<HTMLDivElement>(null)
  const wavesurferRef = useRef<WaveSurfer | null>(null)
  const [isPlaying, setIsPlaying] = useState(false)
  const [isLoading, setIsLoading] = useState(true)
  const [currentTime, setCurrentTime] = useState(0)
  const [duration, setDuration] = useState(0)
  const [volume, setVolume] = useState(1)
  const [isMuted, setIsMuted] = useState(false)

  const formatTime = (time: number) => {
    const minutes = Math.floor(time / 60)
    const seconds = Math.floor(time % 60)
    return `${minutes}:${seconds.toString().padStart(2, '0')}`
  }

  useEffect(() => {
    if (!containerRef.current) return

    const ws = WaveSurfer.create({
      container: containerRef.current,
      waveColor: 'rgba(0, 255, 255, 0.3)',
      progressColor: 'rgba(168, 85, 247, 0.6)',
      cursorColor: 'rgba(0, 255, 255, 1)',
      cursorWidth: 2,
      barWidth: 3,
      barGap: 2,
      barRadius: 3,
      height,
      normalize: true,
      hideScrollbar: true,
      fillParent: true,
      backend: 'WebAudio',
    })

    wavesurferRef.current = ws

    ws.on('ready', () => {
      setIsLoading(false)
      setDuration(ws.getDuration())
      onReady?.()
    })

    ws.on('audioprocess', () => {
      setCurrentTime(ws.getCurrentTime())
    })

    ws.on('seeking', () => {
      setCurrentTime(ws.getCurrentTime())
    })

    ws.on('finish', () => {
      setIsPlaying(false)
      onFinish?.()
    })

    ws.on('play', () => setIsPlaying(true))
    ws.on('pause', () => setIsPlaying(false))

    return () => {
      ws.destroy()
    }
  }, [height, onReady, onFinish])

  useEffect(() => {
    const ws = wavesurferRef.current
    if (!ws) return

    const loadAudio = async () => {
      setIsLoading(true)
      if (audioBlob) {
        const url = URL.createObjectURL(audioBlob)
        await ws.load(url)
      } else if (audioUrl) {
        await ws.load(audioUrl)
      }
    }

    loadAudio()
  }, [audioUrl, audioBlob])

  const togglePlay = useCallback(() => {
    wavesurferRef.current?.playPause()
  }, [])

  const handleVolumeChange = useCallback((value: number[]) => {
    const vol = value[0]
    setVolume(vol)
    wavesurferRef.current?.setVolume(vol)
    if (vol === 0) {
      setIsMuted(true)
    } else {
      setIsMuted(false)
    }
  }, [])

  const toggleMute = useCallback(() => {
    if (isMuted) {
      wavesurferRef.current?.setVolume(volume || 1)
      setIsMuted(false)
    } else {
      wavesurferRef.current?.setVolume(0)
      setIsMuted(true)
    }
  }, [isMuted, volume])

  const restart = useCallback(() => {
    wavesurferRef.current?.seekTo(0)
    wavesurferRef.current?.play()
  }, [])

  return (
    <div className={cn("space-y-3", className)}>
      <div 
        ref={containerRef} 
        className={cn(
          "w-full rounded-lg overflow-hidden",
          isLoading && "animate-pulse bg-cyber-cyan/10"
        )}
      />
      
      <div className="flex items-center gap-4">
        <Button
          variant="ghost"
          size="icon"
          onClick={restart}
          className="h-8 w-8 text-cyber-cyan hover:text-cyber-cyan hover:bg-cyber-cyan/10"
        >
          <SkipBack className="h-4 w-4" />
        </Button>
        
        <Button
          variant="cyber"
          size="icon"
          onClick={togglePlay}
          disabled={isLoading}
          className="h-10 w-10"
        >
          {isPlaying ? (
            <Pause className="h-5 w-5" />
          ) : (
            <Play className="h-5 w-5 ml-0.5" />
          )}
        </Button>

        <div className="flex-1 text-xs text-muted-foreground font-mono">
          {formatTime(currentTime)} / {formatTime(duration)}
        </div>

        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            onClick={toggleMute}
            className="h-8 w-8 text-cyber-cyan hover:text-cyber-cyan hover:bg-cyber-cyan/10"
          >
            {isMuted ? (
              <VolumeX className="h-4 w-4" />
            ) : (
              <Volume2 className="h-4 w-4" />
            )}
          </Button>
          <Slider
            value={[isMuted ? 0 : volume]}
            max={1}
            step={0.01}
            onValueChange={handleVolumeChange}
            className="w-20"
          />
        </div>
      </div>
    </div>
  )
}
