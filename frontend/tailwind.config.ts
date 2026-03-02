import type { Config } from 'tailwindcss'

const config: Config = {
  darkMode: ['class'],
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        border: 'hsl(var(--border))',
        input: 'hsl(var(--input))',
        ring: 'hsl(var(--ring))',
        background: 'hsl(var(--background))',
        foreground: 'hsl(var(--foreground))',
        primary: {
          DEFAULT: 'hsl(var(--primary))',
          foreground: 'hsl(var(--primary-foreground))',
        },
        secondary: {
          DEFAULT: 'hsl(var(--secondary))',
          foreground: 'hsl(var(--secondary-foreground))',
        },
        destructive: {
          DEFAULT: 'hsl(var(--destructive))',
          foreground: 'hsl(var(--destructive-foreground))',
        },
        muted: {
          DEFAULT: 'hsl(var(--muted))',
          foreground: 'hsl(var(--muted-foreground))',
        },
        accent: {
          DEFAULT: 'hsl(var(--accent))',
          foreground: 'hsl(var(--accent-foreground))',
        },
        popover: {
          DEFAULT: 'hsl(var(--popover))',
          foreground: 'hsl(var(--popover-foreground))',
        },
        card: {
          DEFAULT: 'hsl(var(--card))',
          foreground: 'hsl(var(--card-foreground))',
        },
        cyber: {
          cyan: 'hsl(var(--cyber-cyan))',
          purple: 'hsl(var(--cyber-purple))',
          pink: 'hsl(var(--cyber-pink))',
          blue: 'hsl(var(--cyber-blue))',
        },
      },
      borderRadius: {
        lg: 'var(--radius)',
        md: 'calc(var(--radius) - 2px)',
        sm: 'calc(var(--radius) - 4px)',
      },
      fontFamily: {
        sans: ['var(--font-inter)', 'system-ui', 'sans-serif'],
        mono: ['var(--font-jetbrains)', 'monospace'],
      },
      backgroundImage: {
        'gradient-radial': 'radial-gradient(var(--tw-gradient-stops))',
        'cyber-grid': 'linear-gradient(transparent 0%, rgba(0, 255, 255, 0.03) 50%, transparent 100%)',
        'glow-cyan': 'radial-gradient(ellipse at center, rgba(0, 255, 255, 0.15) 0%, transparent 70%)',
        'glow-purple': 'radial-gradient(ellipse at center, rgba(168, 85, 247, 0.15) 0%, transparent 70%)',
      },
      animation: {
        'pulse-glow': 'pulse-glow 2s ease-in-out infinite',
        'float': 'float 6s ease-in-out infinite',
        'shimmer': 'shimmer 2s linear infinite',
        'waveform': 'waveform 1s ease-in-out infinite',
        'glow': 'glow 2s ease-in-out infinite alternate',
        'gradient-x': 'gradient-x 15s ease infinite',
        'spin-slow': 'spin 8s linear infinite',
      },
      keyframes: {
        'pulse-glow': {
          '0%, 100%': { opacity: '1', boxShadow: '0 0 20px rgba(0, 255, 255, 0.5)' },
          '50%': { opacity: '0.8', boxShadow: '0 0 40px rgba(0, 255, 255, 0.8)' },
        },
        'float': {
          '0%, 100%': { transform: 'translateY(0px)' },
          '50%': { transform: 'translateY(-20px)' },
        },
        'shimmer': {
          '0%': { backgroundPosition: '-200% 0' },
          '100%': { backgroundPosition: '200% 0' },
        },
        'waveform': {
          '0%, 100%': { transform: 'scaleY(0.5)' },
          '50%': { transform: 'scaleY(1)' },
        },
        'glow': {
          '0%': { boxShadow: '0 0 20px rgba(0, 255, 255, 0.3)' },
          '100%': { boxShadow: '0 0 30px rgba(0, 255, 255, 0.6), 0 0 60px rgba(168, 85, 247, 0.4)' },
        },
        'gradient-x': {
          '0%, 100%': { backgroundPosition: '0% 50%' },
          '50%': { backgroundPosition: '100% 50%' },
        },
      },
      boxShadow: {
        'cyber': '0 0 20px rgba(0, 255, 255, 0.3), inset 0 0 20px rgba(0, 255, 255, 0.1)',
        'cyber-lg': '0 0 40px rgba(0, 255, 255, 0.4), inset 0 0 40px rgba(0, 255, 255, 0.1)',
        'glow-cyan': '0 0 20px rgba(0, 255, 255, 0.5)',
        'glow-purple': '0 0 20px rgba(168, 85, 247, 0.5)',
      },
    },
  },
  plugins: [require('tailwindcss-animate')],
}

export default config
