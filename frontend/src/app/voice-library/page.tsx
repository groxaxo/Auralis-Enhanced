"use client"

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  FolderOpen, 
  Play, 
  Trash2, 
  Clock, 
  Calendar,
  Volume2,
  Search,
  AlertCircle
} from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { WaveformPlayer } from '@/components/audio/waveform-player'
import { useAppStore } from '@/stores'
import { cn } from '@/lib/utils'

export default function VoiceLibraryPage() {
  const [searchQuery, setSearchQuery] = useState('')
  const [playingVoiceId, setPlayingVoiceId] = useState<string | null>(null)
  
  const { voices, removeVoice } = useAppStore()

  const filteredVoices = voices.filter(voice =>
    voice.name.toLowerCase().includes(searchQuery.toLowerCase())
  )

  const formatDate = (date: Date) => {
    return new Intl.DateTimeFormat('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
    }).format(new Date(date))
  }

  return (
    <div className="max-w-6xl mx-auto space-y-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex items-center gap-3 mb-2">
          <FolderOpen className="h-8 w-8 text-cyber-cyan" />
          <h1 className="text-3xl font-bold text-gradient-cyber">Voice Library</h1>
        </div>
        <p className="text-muted-foreground">
          Manage your saved voice profiles
        </p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 10 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
      >
        <div className="relative">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-muted-foreground" />
          <Input
            placeholder="Search voices..."
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
            className="pl-10"
          />
        </div>
      </motion.div>

      {filteredVoices.length === 0 ? (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
        >
          <Card className="border-dashed">
            <CardContent className="flex flex-col items-center justify-center py-12">
              <Volume2 className="h-12 w-12 text-muted-foreground mb-4" />
              <h3 className="text-lg font-semibold mb-2">No voices saved</h3>
              <p className="text-sm text-muted-foreground text-center max-w-md">
                {searchQuery 
                  ? 'No voices match your search. Try a different query.'
                  : 'Clone and save voices to build your personal voice library.'}
              </p>
              {!searchQuery && (
                <Button variant="cyber" className="mt-4" asChild>
                  <a href="/voice-cloning">Create Voice Profile</a>
                </Button>
              )}
            </CardContent>
          </Card>
        </motion.div>
      ) : (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          transition={{ duration: 0.5, delay: 0.2 }}
          className="grid md:grid-cols-2 lg:grid-cols-3 gap-4"
        >
          <AnimatePresence>
            {filteredVoices.map((voice, index) => (
              <motion.div
                key={voice.id}
                initial={{ opacity: 0, scale: 0.9 }}
                animate={{ opacity: 1, scale: 1 }}
                exit={{ opacity: 0, scale: 0.9 }}
                transition={{ duration: 0.3, delay: index * 0.05 }}
              >
                <Card className="group hover:border-cyber-cyan/30 transition-colors">
                  <CardHeader className="pb-3">
                    <div className="flex items-start justify-between">
                      <div>
                        <CardTitle className="text-lg">{voice.name}</CardTitle>
                        <div className="flex items-center gap-3 mt-1 text-xs text-muted-foreground">
                          <span className="flex items-center gap-1">
                            <Calendar className="h-3 w-3" />
                            {formatDate(voice.createdAt)}
                          </span>
                        </div>
                      </div>
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => removeVoice(voice.id)}
                        className="opacity-0 group-hover:opacity-100 transition-opacity h-8 w-8 text-destructive hover:text-destructive"
                      >
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </CardHeader>
                  <CardContent className="space-y-3">
                    {voice.previewUrl && (
                      <WaveformPlayer
                        audioUrl={voice.previewUrl}
                        height={50}
                      />
                    )}
                    
                    <div className="flex gap-2">
                      <Button
                        variant="outline"
                        size="sm"
                        className="flex-1 gap-2"
                        asChild
                      >
                        <a href={`/?voice=${voice.id}`}>
                          <Play className="h-4 w-4" />
                          Use Voice
                        </a>
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              </motion.div>
            ))}
          </AnimatePresence>
        </motion.div>
      )}

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.3 }}
        className="flex items-center gap-2 text-sm text-muted-foreground"
      >
        <AlertCircle className="h-4 w-4" />
        <span>Voice profiles are stored locally in your browser</span>
      </motion.div>
    </div>
  )
}
