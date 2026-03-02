"use client"

import React, { useCallback, useState, useEffect } from 'react'
import { useDropzone } from 'react-dropzone'
import { Upload, FileAudio, X, CheckCircle2 } from 'lucide-react'
import { motion, AnimatePresence } from 'framer-motion'
import { Button } from '@/components/ui/button'
import { cn } from '@/lib/utils'

interface AudioUploadProps {
  onFilesSelected: (files: File[]) => void
  files: File[]
  maxFiles?: number
  className?: string
}

export function AudioUpload({
  onFilesSelected,
  files,
  maxFiles = 5,
  className,
}: AudioUploadProps) {
  const [showSuccess, setShowSuccess] = useState(false)

  const onDrop = useCallback(
    (acceptedFiles: File[]) => {
      if (acceptedFiles.length > 0) {
        const newFiles = [...files, ...acceptedFiles].slice(0, maxFiles)
        onFilesSelected(newFiles)
        setShowSuccess(true)
      }
    },
    [files, maxFiles, onFilesSelected]
  )

  useEffect(() => {
    if (showSuccess) {
      const timer = setTimeout(() => setShowSuccess(false), 2000)
      return () => clearTimeout(timer)
    }
  }, [showSuccess])

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'audio/*': ['.wav', '.mp3', '.ogg', '.webm', '.m4a', '.flac'],
    },
    maxFiles: maxFiles - files.length,
    disabled: files.length >= maxFiles,
  })

  const removeFile = (index: number) => {
    const newFiles = files.filter((_, i) => i !== index)
    onFilesSelected(newFiles)
  }

  const formatFileSize = (bytes: number) => {
    if (bytes === 0) return '0 Bytes'
    const k = 1024
    const sizes = ['Bytes', 'KB', 'MB', 'GB']
    const i = Math.floor(Math.log(bytes) / Math.log(k))
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
  }

  return (
    <div className={cn('space-y-4', className)}>
      <div
        {...getRootProps()}
        className={cn(
          'border-2 border-dashed rounded-lg p-6 text-center cursor-pointer transition-all duration-300 relative',
          isDragActive
            ? 'border-cyber-cyan bg-cyber-cyan/10 scale-[1.02]'
            : 'border-input hover:border-cyber-cyan/50 hover:bg-accent/5',
          files.length >= maxFiles && 'opacity-50 cursor-not-allowed',
          showSuccess && 'border-green-500 bg-green-500/10'
        )}
      >
        <input {...getInputProps()} />
        
        <AnimatePresence mode="wait">
          {showSuccess ? (
            <motion.div
              key="success"
              initial={{ scale: 0.8, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.8, opacity: 0 }}
              className="flex flex-col items-center"
            >
              <CheckCircle2 className="h-10 w-10 text-green-500 mb-3" />
              <p className="text-green-500 font-medium">File(s) added!</p>
            </motion.div>
          ) : (
            <motion.div
              key="upload"
              initial={{ scale: 0.95, opacity: 0 }}
              animate={{ scale: 1, opacity: 1 }}
              exit={{ scale: 0.95, opacity: 0 }}
            >
              <Upload
                className={cn(
                  'mx-auto h-10 w-10 mb-3 transition-colors',
                  isDragActive ? 'text-cyber-cyan' : 'text-muted-foreground'
                )}
              />
              {isDragActive ? (
                <p className="text-cyber-cyan font-medium">Drop audio files here...</p>
              ) : (
                <div className="space-y-1">
                  <p className="text-sm text-foreground">
                    Drag & drop audio files, or click to select
                  </p>
                  <p className="text-xs text-muted-foreground">
                    WAV, MP3, OGG, WebM, M4A, FLAC ({maxFiles - files.length} slots remaining)
                  </p>
                </div>
              )}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      <AnimatePresence>
        {files.length > 0 && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="space-y-2"
          >
            <p className="text-sm text-muted-foreground">
              {files.length} file{files.length !== 1 ? 's' : ''} selected:
            </p>
            {files.map((file, index) => (
              <motion.div
                key={`${file.name}-${index}`}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                exit={{ opacity: 0, x: 20 }}
                transition={{ delay: index * 0.05 }}
                className="flex items-center gap-3 p-3 rounded-lg bg-accent/10 border border-accent/20 group"
              >
                <FileAudio className="h-5 w-5 text-cyber-cyan flex-shrink-0" />
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium truncate">{file.name}</p>
                  <p className="text-xs text-muted-foreground">
                    {formatFileSize(file.size)}
                  </p>
                </div>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={(e) => {
                    e.stopPropagation()
                    removeFile(index)
                  }}
                  className="h-8 w-8 hover:bg-destructive/20 hover:text-destructive"
                >
                  <X className="h-4 w-4" />
                </Button>
              </motion.div>
            ))}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  )
}
