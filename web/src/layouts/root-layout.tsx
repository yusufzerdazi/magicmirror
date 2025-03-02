import { Outlet } from 'react-router-dom';

export default function RootLayout() {
  const renderContent = () => (
    <div className="min-h-screen bg-stone-900 text-white flex flex-col">
      <main>
        <Outlet />
      </main>
    </div>
  );

  return renderContent();
}
