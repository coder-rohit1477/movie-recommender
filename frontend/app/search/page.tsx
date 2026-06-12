"use client";

import { useSearchParams } from 'next/navigation';
import { useQuery } from '@tanstack/react-query';
import api from '@/lib/axios';
import Navbar from '@/components/Navbar';
import MovieCard from '@/components/MovieCard';

export default function SearchPage() {
  const searchParams = useSearchParams();
  const query = searchParams.get('q');

  const { data: results, isLoading } = useQuery({
    queryKey: ['search', query],
    queryFn: async () => {
      if (!query) return [];
      const response = await api.get(`/movies/search?q=${query}`);
      return response.data.results || [];
    },
    enabled: !!query,
  });

  return (
    <main className="min-h-screen pb-20">
      <Navbar />
      
      <div className="pt-24 px-4 md:px-12 space-y-8">
        <h2 className="text-xl md:text-2xl font-medium text-gray-400">
          Showing results for: <span className="text-white font-bold">{query}</span>
        </h2>

        {isLoading ? (
          <div className="flex items-center justify-center h-60">
            <div className="animate-spin rounded-full h-12 w-12 border-t-2 border-b-2 border-netflix-red"></div>
          </div>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-x-2 gap-y-8">
            {results.map((title: string, index: number) => (
              <div key={index} className="flex flex-col items-center">
                <MovieCard title={title} />
              </div>
            ))}
            
            {results.length === 0 && !isLoading && (
              <p className="text-gray-500 col-span-full text-center py-20">
                No movies found matching your search.
              </p>
            )}
          </div>
        )}
      </div>
    </main>
  );
}
