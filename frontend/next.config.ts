import type { NextConfig } from 'next'

const nextConfig: NextConfig = {
  reactStrictMode: true,
  output: 'standalone',
  experimental: {
    optimizePackageImports: ['lucide-react', '@radix-ui/react-icons'],
  },
  allowedDevOrigins: ['localhost', '.tailscale.com', '100.85.200.51'],
}

export default nextConfig
