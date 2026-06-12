"use client";

import React from 'react';
import { Play, Info } from 'lucide-react';

const HeroBanner = () => {
  return (
    <div className="relative h-[85vh] w-full">
      <div className="absolute inset-0">
        <img 
          src="https://wallpapers.com/images/hd/movie-poster-background-1920-x-1080-92a83h745g732111.jpg" 
          className="w-full h-full object-cover" 
          alt="Hero" 
        />
        <div className="absolute inset-0 bg-gradient-to-r from-netflix-black via-transparent to-transparent" />
        <div className="absolute inset-0 bg-gradient-to-t from-netflix-black via-transparent to-transparent" />
      </div>

      <div className="absolute bottom-24 left-4 md:left-12 max-w-xl space-y-4 px-4">
        <h2 className="text-4xl md:text-7xl font-bold tracking-tight">AI FOR YOU</h2>
        <p className="text-lg md:text-xl font-medium text-gray-200 drop-shadow-md">
          Discover your next favorite story. Our hybrid engine (SVD + FAISS) 
          analyzes millions of interactions to bring you the best content from MovieLens.
        </p>
        <div className="flex gap-4 pt-4">
          <button className="flex items-center gap-2 bg-white text-black px-8 py-2.5 rounded font-bold hover:bg-gray-200 transition text-lg">
            <Play className="fill-black" /> Play
          </button>
          <button className="flex items-center gap-2 bg-gray-500/60 text-white px-8 py-2.5 rounded font-bold hover:bg-gray-500/40 transition text-lg">
            <Info /> More Info
          </button>
        </div>
      </div>
    </div>
  );
};

export default HeroBanner;
