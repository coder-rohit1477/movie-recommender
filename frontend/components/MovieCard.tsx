"use client";

import React, { useState } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Star, Play, Plus, ThumbsUp } from 'lucide-react';

interface MovieCardProps {
  title: string;
  score?: number;
  year?: string;
}

const MovieCard: React.FC<MovieCardProps> = ({ title, score, year }) => {
  const [isHovered, setIsHovered] = useState(false);
  
  // Extract base title for cleaner placeholder text
  const baseTitle = title.split(' (')[0];
  const movieYear = year || (title.match(/\((\d{4})\)/)?.[1]);
  
  // Using placehold.co as a fallback, TMDB would be better in production
  const posterUrl = `https://placehold.co/300x450/222/FFF?text=${encodeURIComponent(baseTitle)}`;

  return (
    <motion.div
      className="relative flex-shrink-0 w-40 md:w-56 h-28 md:h-32 bg-[#222] rounded-sm overflow-hidden cursor-pointer"
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      whileHover={{ scale: 1.1, zIndex: 50 }}
      transition={{ duration: 0.2 }}
    >
      <img 
        src={posterUrl} 
        alt={title} 
        className="w-full h-full object-cover"
      />
      
      <AnimatePresence>
        {isHovered && (
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="absolute inset-0 bg-black/70 p-3 flex flex-col justify-between"
          >
            <div>
              <h4 className="text-xs md:text-sm font-bold line-clamp-2">{title}</h4>
              <div className="flex items-center gap-2 mt-1">
                <div className="flex items-center gap-1">
                  <Star className="w-3 h-3 text-green-500 fill-green-500" />
                  <span className="text-[10px] md:text-xs text-green-500 font-bold">
                    {score ? Math.round(score * 100) : 98}% Match
                  </span>
                </div>
                {movieYear && <span className="text-[10px] md:text-xs text-gray-400">{movieYear}</span>}
              </div>
            </div>
            
            <div className="flex gap-2">
              <button className="p-1.5 bg-white rounded-full hover:bg-gray-200 transition">
                <Play size={12} className="fill-black text-black ml-0.5" />
              </button>
              <button className="p-1.5 border border-gray-400 rounded-full hover:border-white transition">
                <Plus size={12} />
              </button>
              <button className="p-1.5 border border-gray-400 rounded-full hover:border-white transition ml-auto">
                <ThumbsUp size={12} />
              </button>
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </motion.div>
  );
};

export default MovieCard;
