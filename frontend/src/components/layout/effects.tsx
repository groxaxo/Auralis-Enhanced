"use client"

import React from 'react'
import { motion } from 'framer-motion'

export function BackgroundEffects() {
  return (
    <div className="fixed inset-0 overflow-hidden pointer-events-none">
      <div className="absolute inset-0 bg-gradient-radial from-cyber-cyan/5 via-transparent to-transparent" />
      
      <div className="absolute top-0 left-1/4 w-96 h-96 bg-cyber-cyan/10 rounded-full blur-3xl animate-float" />
      <div 
        className="absolute bottom-1/4 right-1/4 w-80 h-80 bg-cyber-purple/10 rounded-full blur-3xl animate-float" 
        style={{ animationDelay: '-3s' }}
      />
      <div 
        className="absolute top-1/2 left-1/2 w-64 h-64 bg-cyber-pink/5 rounded-full blur-3xl animate-float"
        style={{ animationDelay: '-1.5s' }}
      />

      <div className="absolute inset-0 bg-noise" />

      <svg className="absolute inset-0 w-full h-full opacity-5">
        <defs>
          <pattern id="grid" width="40" height="40" patternUnits="userSpaceOnUse">
            <path 
              d="M 40 0 L 0 0 0 40" 
              fill="none" 
              stroke="currentColor"
              strokeWidth="0.5"
              className="text-cyber-cyan"
            />
          </pattern>
        </defs>
        <rect width="100%" height="100%" fill="url(#grid)" />
      </svg>
    </div>
  )
}

export function LoadingSpinner({ className }: { className?: string }) {
  return (
    <div className={className}>
      <motion.div
        animate={{ rotate: 360 }}
        transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
        className="w-8 h-8 rounded-full border-2 border-cyber-cyan border-t-transparent"
      />
    </div>
  )
}

export function GlowOrb({ 
  color = 'cyan', 
  size = 'md',
  className 
}: { 
  color?: 'cyan' | 'purple' | 'pink'
  size?: 'sm' | 'md' | 'lg'
  className?: string 
}) {
  const colors = {
    cyan: 'bg-cyber-cyan',
    purple: 'bg-cyber-purple',
    pink: 'bg-cyber-pink',
  }
  
  const sizes = {
    sm: 'w-2 h-2',
    md: 'w-3 h-3',
    lg: 'w-4 h-4',
  }
  
  return (
    <motion.div
      animate={{ 
        scale: [1, 1.2, 1],
        opacity: [0.5, 1, 0.5]
      }}
      transition={{ 
        duration: 2,
        repeat: Infinity,
        ease: 'easeInOut'
      }}
      className={`${sizes[size]} ${colors[color]} rounded-full ${className}`}
      style={{
        boxShadow: `0 0 20px currentColor`,
      }}
    />
  )
}
