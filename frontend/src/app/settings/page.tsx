"use client"

import React, { useState } from 'react'
import { motion } from 'framer-motion'
import { Settings, Server, Palette, Database, Check, AlertCircle } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from '@/components/ui/card'
import { Input } from '@/components/ui/input'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { useAppStore } from '@/stores'
import { auralisAPI } from '@/lib/api'
import { cn } from '@/lib/utils'

export default function SettingsPage() {
  const [testStatus, setTestStatus] = useState<'idle' | 'testing' | 'success' | 'error'>('idle')
  const { apiBaseUrl, setApiBaseUrl, theme, setTheme } = useAppStore()
  const [localApiUrl, setLocalApiUrl] = useState(apiBaseUrl)

  const handleTestConnection = async () => {
    setTestStatus('testing')
    auralisAPI.setBaseUrl(localApiUrl)
    
    const isHealthy = await auralisAPI.healthCheck()
    setTestStatus(isHealthy ? 'success' : 'error')
    
    if (isHealthy) {
      setApiBaseUrl(localApiUrl)
    }
  }

  const handleSaveApiUrl = () => {
    setApiBaseUrl(localApiUrl)
  }

  return (
    <div className="max-w-3xl mx-auto space-y-6">
      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5 }}
      >
        <div className="flex items-center gap-3 mb-2">
          <Settings className="h-8 w-8 text-cyber-cyan" />
          <h1 className="text-3xl font-bold text-gradient-cyber">Settings</h1>
        </div>
        <p className="text-muted-foreground">
          Configure your Auralis Enhanced experience
        </p>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.1 }}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Server className="h-5 w-5 text-cyber-cyan" />
              API Configuration
            </CardTitle>
            <CardDescription>
              Connect to the Auralis backend server
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="space-y-2">
              <Label htmlFor="api-url">Backend URL</Label>
              <div className="flex gap-2">
                <Input
                  id="api-url"
                  placeholder="http://localhost:8000"
                  value={localApiUrl}
                  onChange={(e) => {
                    setLocalApiUrl(e.target.value)
                    setTestStatus('idle')
                  }}
                  className="flex-1"
                />
                <Button
                  variant="outline"
                  onClick={handleTestConnection}
                  disabled={testStatus === 'testing'}
                >
                  {testStatus === 'testing' ? (
                    <motion.div
                      animate={{ rotate: 360 }}
                      transition={{ duration: 1, repeat: Infinity, ease: 'linear' }}
                      className="w-4 h-4 border-2 border-cyber-cyan border-t-transparent rounded-full"
                    />
                  ) : testStatus === 'success' ? (
                    <Check className="h-4 w-4 text-green-500" />
                  ) : testStatus === 'error' ? (
                    <AlertCircle className="h-4 w-4 text-destructive" />
                  ) : (
                    'Test'
                  )}
                </Button>
              </div>
              {testStatus === 'success' && (
                <p className="text-sm text-green-500 flex items-center gap-1">
                  <Check className="h-3 w-3" />
                  Connection successful
                </p>
              )}
              {testStatus === 'error' && (
                <p className="text-sm text-destructive flex items-center gap-1">
                  <AlertCircle className="h-3 w-3" />
                  Could not connect to server
                </p>
              )}
            </div>
            
            <div className="flex gap-2">
              <Button variant="cyber" onClick={handleSaveApiUrl}>
                Save Configuration
              </Button>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.2 }}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Palette className="h-5 w-5 text-cyber-purple" />
              Appearance
            </CardTitle>
            <CardDescription>
              Customize the look and feel
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between">
              <div>
                <Label>Dark Mode</Label>
                <p className="text-sm text-muted-foreground">
                  Use dark theme (recommended)
                </p>
              </div>
              <Switch
                checked={theme === 'dark'}
                onCheckedChange={(checked) => setTheme(checked ? 'dark' : 'light')}
              />
            </div>
          </CardContent>
        </Card>
      </motion.div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.5, delay: 0.3 }}
      >
        <Card>
          <CardHeader>
            <CardTitle className="text-lg flex items-center gap-2">
              <Database className="h-5 w-5 text-cyber-cyan" />
              Data Management
            </CardTitle>
            <CardDescription>
              Manage your local data and storage
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex items-center justify-between p-3 rounded-lg bg-accent/10">
              <div>
                <p className="font-medium">Clear All Data</p>
                <p className="text-sm text-muted-foreground">
                  Remove all saved voices and presets
                </p>
              </div>
              <Button
                variant="destructive"
                size="sm"
                onClick={() => {
                  if (confirm('Are you sure? This will delete all your saved voices and presets.')) {
                    localStorage.removeItem('auralis-storage')
                    window.location.reload()
                  }
                }}
              >
                Clear Data
              </Button>
            </div>
          </CardContent>
        </Card>
      </motion.div>

      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        transition={{ duration: 0.5, delay: 0.4 }}
        className="text-sm text-muted-foreground text-center"
      >
        Auralis Enhanced v1.0.0 • Built with Next.js & Tailwind CSS
      </motion.div>
    </div>
  )
}
