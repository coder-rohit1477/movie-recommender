"use client";

import React, { useRef } from 'react';
import { useQuery } from '@tanstack/react-query';
import api from '@/lib/axios';
import MovieCard from './MovieCard';
import { ChevronLeft, ChevronRight, AlertCircle } from 'lucide-react';

interface MovieRowProps {
  title: string;
  endpoint: string;
}

const MovieRowSkeleton = () => (
  <div className="py-4 md:py-8 space-y-4 px-4 md:px-12">
    <div className="h-8 w-48 bg-gray-800 animate-pulse rounded" />
    <div className="flex gap-2 overflow-hidden">
      {[...Array(6)].map((_, i) => (
        <div key={i} className="flex-shrink-0 w-40 md:w-56 h-28 md:h-32 bg-gray-800 animate-pulse rounded-sm" />
      ))}
    </div>
  </div>
);

const MovieRow: React.FC<MovieRowProps> = ({ title, endpoint }) => {
  const scrollRef = useRef<HTMLDivElement>(null);

  const { data: movies, isLoading, isError, error } = useQuery({
    queryKey: [endpoint],
    queryFn: async () => {
      try {
        const response = await api.get(endpoint);
        return response.data.recommendations || response.data.movies || response.data.results || [];
      } catch (err: any) {
        console.error(`Error fetching from ${endpoint}:`, err);
        throw err;
      }
    },
    retry: 1,
  });

  const scroll = (direction: 'left' | 'right') => {
    if (scrollRef.current) {
      const { scrollLeft, clientWidth } = scrollRef.current;
      const scrollTo = direction === 'left' ? scrollLeft - clientWidth : scrollLeft + clientWidth;
      scrollRef.current.scrollTo({ left: scrollTo, behavior: 'smooth' });
    }
  };

  if (isLoading) return <MovieRowSkeleton />;

  if (isError) {
    return (
      <div className="py-4 md:py-8 px-4 md:px-12 text-gray-500 flex items-center gap-2">
        <AlertCircle size={20} />
        <span>Failed to load {title.toLowerCase()}</span>
      </div>
    );
  }

  if (!movies || movies.length === 0) return null;

  return (
    <div className="py-4 md:py-8 space-y-4 px-4 md:px-12 group overflow-hidden">
      <h3 className="text-lg md:text-2xl font-bold text-gray-200 hover:text-white cursor-pointer transition">
        {title}
      </h3>
      
      <div className="relative flex items-center">
        <div 
          className="absolute left-0 z-10 p-2 bg-black/40 hover:bg-black/60 cursor-pointer opacity-0 group-hover:opacity-100 transition rounded-full ml-1" 
          onClick={() => scroll('left')}
        >
          <ChevronLeft className="w-8 h-8" />
        </div>

        <div 
          ref={scrollRef} 
          className="flex gap-2 overflow-x-auto scrollbar-hide scroll-smooth py-4"
        >
          {movies.map((movie: any, index: number) => (
            <MovieCard 
              key={index} 
              title={typeof movie === 'string' ? movie : movie.title} 
              score={movie.score}
              year={movie.year}
            />
          ))}
        </div>

        <div 
          className="absolute right-0 z-10 p-2 bg-black/40 hover:bg-black/60 cursor-pointer opacity-0 group-hover:opacity-100 transition rounded-full mr-1" 
          onClick={() => scroll('right')}
        >
          <ChevronRight className="w-8 h-8" />
        </div>
      </div>
    </div>
  );
};

export default MovieRow;
