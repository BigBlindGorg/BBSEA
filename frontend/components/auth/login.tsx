'use client';

import { useState } from 'react';
import Image from 'next/image';
import { Button } from '@/components/ui/button';

interface LoginProps {
  onLogin: (success: boolean) => void;
}

export const Login = ({ onLogin }: LoginProps) => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();

    const validUsers = [
      { email: 'user1@example.com', password: 'your-password-here' },
      { email: 'user2@example.com', password: 'your-password-here' },
      { email: 'user3@example.com', password: 'your-password-here' }
    ];

    const isValid = validUsers.some(
      user => user.email === email && user.password === password
    );

    if (isValid) {
      onLogin(true);
    } else {
      setError('Invalid credentials');
      setEmail('');
      setPassword('');
    }
  };

  return (
    <div className="flex h-screen w-full items-center justify-center bg-[#191a1a] -mt-32">
      <div className="w-full max-w-md space-y-2 px-4">
        <div className="flex flex-col items-center">
          <Image
            src="/logo.png"
            alt="Queen-RAG"
            width={600}
            height={600}
            className="h-64 w-auto"
            priority
          />
          <h2 className="mt-2 text-center text-3xl font-regular text-foreground">
            Chat with NEUVIA application documents
          </h2>
        </div>

        <form onSubmit={handleSubmit} className="mt-3 space-y-6">
          <div className="space-y-4 rounded-lg bg-[#202222] p-6 border border-white/10">
            <div>
              <label htmlFor="email" className="block text-sm font-medium text-foreground mb-2">
                Email address
              </label>
              <input
                id="email"
                name="email"
                type="email"
                autoComplete="email"
                required
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full rounded-lg border border-white/10 bg-[#191a1a] px-4 py-3 text-foreground placeholder:text-muted-foreground focus:border-white/20 focus:outline-none focus:ring-1 focus:ring-white/20 transition-all"
                placeholder="Enter your email"
              />
            </div>

            <div>
              <label htmlFor="password" className="block text-sm font-medium text-foreground mb-2">
                Password
              </label>
              <input
                id="password"
                name="password"
                type="password"
                autoComplete="current-password"
                required
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                className="w-full rounded-lg border border-white/10 bg-[#191a1a] px-4 py-3 text-foreground placeholder:text-muted-foreground focus:border-white/20 focus:outline-none focus:ring-1 focus:ring-white/20 transition-all"
                placeholder="Enter your password"
              />
            </div>

            {error && (
              <div className="rounded-lg bg-red-500/10 border border-red-500/20 px-4 py-3 text-sm text-red-500">
                {error}
              </div>
            )}

            <Button
              type="submit"
              className="w-full rounded-lg bg-blue-600 px-4 py-3 text-white font-medium hover:bg-blue-700 transition-all"
            >
              Sign in
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
};
