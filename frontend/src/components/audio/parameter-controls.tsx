"use client"

import React from 'react'
import { Slider } from '@/components/ui/slider'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { cn } from '@/lib/utils'

interface ParameterSliderProps {
  label: string
  value: number
  onChange: (value: number) => void
  min: number
  max: number
  step: number
  description?: string
  unit?: string
  className?: string
}

export function ParameterSlider({
  label,
  value,
  onChange,
  min,
  max,
  step,
  description,
  unit,
  className,
}: ParameterSliderProps) {
  return (
    <div className={cn('space-y-2', className)}>
      <div className="flex items-center justify-between">
        <Label className="text-sm">{label}</Label>
        <span className="text-sm font-mono text-cyber-cyan">
          {value.toFixed(step < 1 ? 2 : 0)}
          {unit && <span className="text-muted-foreground ml-1">{unit}</span>}
        </span>
      </div>
      <Slider
        value={[value]}
        onValueChange={([v]) => onChange(v)}
        min={min}
        max={max}
        step={step}
      />
      {description && (
        <p className="text-xs text-muted-foreground">{description}</p>
      )}
    </div>
  )
}

interface ParameterToggleProps {
  label: string
  checked: boolean
  onChange: (checked: boolean) => void
  description?: string
  badge?: string
  className?: string
}

export function ParameterToggle({
  label,
  checked,
  onChange,
  description,
  badge,
  className,
}: ParameterToggleProps) {
  return (
    <div className={cn('flex items-start justify-between gap-4', className)}>
      <div className="space-y-0.5 flex-1">
        <div className="flex items-center gap-2">
          <Label className="text-sm cursor-pointer">{label}</Label>
          {badge && (
            <span className="text-xs px-2 py-0.5 rounded-full bg-gradient-to-r from-cyber-cyan/20 to-cyber-purple/20 text-cyber-cyan border border-cyber-cyan/30">
              {badge}
            </span>
          )}
        </div>
        {description && (
          <p className="text-xs text-muted-foreground">{description}</p>
        )}
      </div>
      <Switch checked={checked} onCheckedChange={onChange} />
    </div>
  )
}
