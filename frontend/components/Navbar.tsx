"use client";

import React, { useState, useEffect } from 'react';
import Link from 'next/link';
import { useRouter } from 'next/navigation';
import { Search, Bell, User } from 'lucide-react';
import { motion, AnimatePresence } from 'framer-motion';
import api from '@/lib/axios';

const Navbar = () => {
  const [isScrolled, setIsScrolled] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [showSearch, setShowSearch] = useState(false);
  const [results, setResults] = useState<string[]>([]);
  const router = useRouter();

  useEffect(() => {
    const handleScroll = () => {
      setIsScrolled(window.scrollY > 0);
    };
    window.addEventListener('scroll', handleScroll);
    return () => window.removeEventListener('scroll', handleScroll);
  }, []);

  useEffect(() => {
    if (searchQuery.length > 2) {
      const delay = setTimeout(async () => {
        try {
          const res = await api.get(`/movies/search?q=${searchQuery}`);
          setResults(res.data.results || []);
        } catch (e) {
          setResults([]);
        }
      }, 300);
      return () => clearTimeout(delay);
    } else {
      setResults([]);
    }
  }, [searchQuery]);

  const handleSearch = (e: React.FormEvent) => {
    e.preventDefault();
    if (searchQuery) {
      router.push(`/search?q=${encodeURIComponent(searchQuery)}`);
      setShowSearch(false);
    }
  };

  return (
    <nav className={`fixed top-0 w-full z-[100] transition-all duration-500 px-4 md:px-12 py-4 flex items-center justify-between ${isScrolled ? 'bg-netflix-black shadow-lg' : 'bg-gradient-to-b from-black/80 to-transparent'}`}>
      <div className="flex items-center gap-8">
        <Link href="/home">
          <h1 className="text-netflix-red text-2xl md:text-3xl font-extrabold tracking-tighter cursor-pointer">STREAMFLIX</h1>
        </Link>
        <div className="hidden lg:flex gap-5 text-sm font-medium text-gray-300">
          <Link href="/home" className="hover:text-white transition">Home</Link>
          <span className="hover:text-white cursor-pointer transition">TV Shows</span>
          <span className="hover:text-white cursor-pointer transition">Movies</span>
          <span className="hover:text-white cursor-pointer transition">New & Popular</span>
          <span className="hover:text-white cursor-pointer transition">My List</span>
        </div>
      </div>

      <div className="flex items-center gap-6">
        <div className="relative flex items-center">
          <motion.form 
            onSubmit={handleSearch}
            animate={{ width: showSearch ? 260 : 40 }} 
            className="flex items-center bg-black/40 border border-gray-600 rounded overflow-hidden"
          >
            <Search 
              className="w-5 h-5 ml-2 cursor-pointer text-gray-400" 
              onClick={() => setShowSearch(!showSearch)} 
            />
            <input 
              className={`bg-transparent border-none outline-none text-sm p-2 transition-all text-white w-full ${showSearch ? 'opacity-100' : 'opacity-0'}`}
              placeholder="Titles, people, genres"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              autoFocus={showSearch}
            />
          </motion.form>
          
          <AnimatePresence>
            {results.length > 0 && showSearch && (
              <motion.div 
                initial={{ opacity: 0, y: -10 }} 
                animate={{ opacity: 1, y: 0 }} 
                exit={{ opacity: 0 }} 
                className="absolute top-full left-0 w-full bg-netflix-black border border-gray-700 mt-2 rounded shadow-2xl max-h-60 overflow-y-auto"
              >
                {results.map((title, i) => (
                  <div 
                    key={i} 
                    className="px-4 py-2 hover:bg-gray-800 cursor-pointer text-sm border-b border-gray-800 last:border-0"
                    onClick={() => {
                      router.push(`/search?q=${encodeURIComponent(title)}`);
                      setShowSearch(false);
                      setSearchQuery('');
                    }}
                  >
                    {title}
                  </div>
                ))}
              </motion.div>
            )}
          </AnimatePresence>
        </div>
        
        <Bell className="w-6 h-6 text-gray-300 hover:text-white cursor-pointer transition hidden sm:block" />
        <div className="w-8 h-8 rounded bg-gray-600 flex items-center justify-center cursor-pointer">
          <User className="w-5 h-5" />
        </div>
      </div>
    </nav>
  );
};

export default Navbar;
