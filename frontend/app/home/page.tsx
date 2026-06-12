import Navbar from '@/components/Navbar';
import HeroBanner from '@/components/HeroBanner';
import MovieRow from '@/components/MovieRow';

export default function HomePage() {
  // We'll use a hardcoded user_id for the demo, 
  // in a real app this would come from an auth context.
  const USER_ID = "196";

  return (
    <main className="min-h-screen pb-20 overflow-x-hidden">
      <Navbar />
      <HeroBanner />
      
      <div className="-mt-16 md:-mt-32 relative z-20 space-y-4 md:space-y-8">
        <MovieRow 
          title="Recommended For You" 
          endpoint={`/recommend/hybrid?user_id=${USER_ID}`} 
        />
        <MovieRow 
          title="Trending Now" 
          endpoint="/recommend/trending" 
        />
        <MovieRow 
          title="Popular Movies" 
          endpoint={`/recommend/popular?user_id=${USER_ID}`} 
        />
        <MovieRow 
          title="Top Picks in Sci-Fi" 
          endpoint="/recommend/hybrid?user_id=196&movie_title=Star Wars (1977)" 
        />
      </div>
    </main>
  );
}
