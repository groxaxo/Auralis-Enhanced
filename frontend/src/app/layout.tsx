import type { Metadata } from 'next'
import { Inter, JetBrains_Mono } from 'next/font/google'
import './globals.css'
import { Sidebar } from '@/components/layout/sidebar'
import { BackgroundEffects } from '@/components/layout/effects'

const inter = Inter({ 
  subsets: ['latin'],
  variable: '--font-inter',
})

const jetbrainsMono = JetBrains_Mono({
  subsets: ['latin'],
  variable: '--font-jetbrains',
})

export const metadata: Metadata = {
  title: 'Auralis Enhanced - Futuristic TTS',
  description: 'Professional text-to-speech with voice cloning and NovaSR audio enhancement',
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" className="dark">
      <body className={`${inter.variable} ${jetbrainsMono.variable} font-sans antialiased`}>
        <BackgroundEffects />
        <div className="flex min-h-screen">
          <Sidebar />
          <main className="flex-1 lg:ml-64 p-4 lg:p-8 pt-20 lg:pt-8">
            {children}
          </main>
        </div>
      </body>
    </html>
  )
}
