'use client';

import { useState, useEffect } from 'react';
import { Perplexity } from '@/components/perplexity/Perplexity';
import { AssistantProvider } from './assistant-provider';
import { Login } from '@/components/auth/login';

export default function Home() {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    // Check if user is already authenticated
    const authStatus = localStorage.getItem('neuvia_auth');
    if (authStatus === 'true') {
      setIsAuthenticated(true);
    }
    setIsLoading(false);
  }, []);

  const handleLogin = (success: boolean) => {
    if (success) {
      localStorage.setItem('neuvia_auth', 'true');
      setIsAuthenticated(true);
    }
  };

  if (isLoading) {
    return null;
  }

  if (!isAuthenticated) {
    return <Login onLogin={handleLogin} />;
  }

  return (
    <AssistantProvider>
      <Perplexity />
    </AssistantProvider>
  );
}