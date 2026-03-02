"use client"

import React, { useState } from 'react'
import Link from 'next/link'
import { usePathname } from 'next/navigation'
import { motion } from 'framer-motion'
import { 
  AudioWaveform, 
  Mic, 
  FolderOpen, 
  Settings, 
  Sparkles,
  Menu,
  X
} from 'lucide-react'
import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'

const navItems = [
  { href: '/', label: 'Generate', icon: AudioWaveform },
  { href: '/voice-cloning', label: 'Voice Clone', icon: Mic },
  { href: '/voice-library', label: 'Voice Library', icon: FolderOpen },
  { href: '/settings', label: 'Settings', icon: Settings },
]

export function Sidebar() {
  const pathname = usePathname()
  const [isOpen, setIsOpen] = useState(false)

  return (
    <>
      <motion.aside
        initial={{ x: -100, opacity: 0 }}
        animate={{ x: 0, opacity: 1 }}
        transition={{ duration: 0.5, ease: 'easeOut' }}
        className={cn(
          "fixed left-0 top-0 z-40 h-screen w-64 glass border-r border-cyber-cyan/20",
          "hidden lg:flex flex-col",
          "transition-transform duration-300"
        )}
      >
        <div className="flex items-center gap-3 p-6 border-b border-cyber-cyan/10">
          <div className="relative">
            <Sparkles className="h-8 w-8 text-cyber-cyan animate-pulse" />
            <div className="absolute inset-0 blur-lg bg-cyber-cyan/50 animate-pulse" />
          </div>
          <div>
            <h1 className="text-xl font-bold text-gradient-cyber">Auralis</h1>
            <p className="text-xs text-muted-foreground">Enhanced TTS</p>
          </div>
        </div>

        <nav className="flex-1 p-4 space-y-2">
          {navItems.map((item) => {
            const isActive = pathname === item.href
            const Icon = item.icon
            
            return (
              <Link key={item.href} href={item.href}>
                <motion.div
                  whileHover={{ x: 4 }}
                  whileTap={{ scale: 0.98 }}
                  className={cn(
                    "flex items-center gap-3 px-4 py-3 rounded-lg transition-all duration-300",
                    "hover:bg-cyber-cyan/10",
                    isActive && "bg-gradient-to-r from-cyber-cyan/20 to-cyber-purple/20 border border-cyber-cyan/30"
                  )}
                >
                  <Icon className={cn(
                    "h-5 w-5 transition-colors",
                    isActive ? "text-cyber-cyan" : "text-muted-foreground"
                  )} />
                  <span className={cn(
                    "font-medium transition-colors",
                    isActive ? "text-foreground" : "text-muted-foreground"
                  )}>
                    {item.label}
                  </span>
                  {isActive && (
                    <motion.div
                      layoutId="activeIndicator"
                      className="ml-auto w-2 h-2 rounded-full bg-cyber-cyan shadow-glow-cyan"
                    />
                  )}
                </motion.div>
              </Link>
            )
          })}
        </nav>

        <div className="p-4 border-t border-cyber-cyan/10">
          <div className="flex items-center gap-2 px-4 py-2 text-xs text-muted-foreground">
            <div className="w-2 h-2 rounded-full bg-green-500 animate-pulse" />
            <span>System Ready</span>
          </div>
        </div>
      </motion.aside>

      <div className="lg:hidden fixed top-0 left-0 right-0 z-50 glass border-b border-cyber-cyan/20">
        <div className="flex items-center justify-between p-4">
          <div className="flex items-center gap-2">
            <Sparkles className="h-6 w-6 text-cyber-cyan" />
            <span className="font-bold text-gradient-cyber">Auralis</span>
          </div>
          <Button
            variant="ghost"
            size="icon"
            onClick={() => setIsOpen(!isOpen)}
          >
            {isOpen ? <X className="h-5 w-5" /> : <Menu className="h-5 w-5" />}
          </Button>
        </div>
      </div>

      {isOpen && (
        <motion.div
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          className="lg:hidden fixed inset-0 z-40 bg-background/80 backdrop-blur-sm"
          onClick={() => setIsOpen(false)}
        >
          <motion.nav
            initial={{ x: '100%' }}
            animate={{ x: 0 }}
            exit={{ x: '100%' }}
            transition={{ type: 'spring', damping: 20 }}
            className="absolute right-0 top-16 bottom-0 w-64 glass border-l border-cyber-cyan/20 p-4 space-y-2"
            onClick={(e) => e.stopPropagation()}
          >
            {navItems.map((item) => {
              const isActive = pathname === item.href
              const Icon = item.icon
              
              return (
                <Link key={item.href} href={item.href} onClick={() => setIsOpen(false)}>
                  <div className={cn(
                    "flex items-center gap-3 px-4 py-3 rounded-lg transition-all",
                    isActive 
                      ? "bg-gradient-to-r from-cyber-cyan/20 to-cyber-purple/20 border border-cyber-cyan/30" 
                      : "hover:bg-cyber-cyan/10"
                  )}>
                    <Icon className={cn(
                      "h-5 w-5",
                      isActive ? "text-cyber-cyan" : "text-muted-foreground"
                    )} />
                    <span className={cn(
                      "font-medium",
                      isActive ? "text-foreground" : "text-muted-foreground"
                    )}>
                      {item.label}
                    </span>
                  </div>
                </Link>
              )
            })}
          </motion.nav>
        </motion.div>
      )}
    </>
  )
}
