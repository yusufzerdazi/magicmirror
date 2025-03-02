import React, { useState } from 'react';

interface PasswordAuthProps {
  onAuthenticated: (password: string) => void;
}

const PasswordAuth: React.FC<PasswordAuthProps> = ({ onAuthenticated }) => {
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsLoading(true);
    setError('');

    try {
      const response = await fetch('http://192.168.1.113:5556/auth/verify', {
        method: 'POST',
        headers: {
          'Authorization': `Bearer ${password}`,
          'Content-Type': 'application/json'
        }
      });

      if (response.ok) {
        onAuthenticated(password);
      } else {
        setError('Invalid password');
      }
    } catch (err) {
      setError('Failed to authenticate. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="fixed inset-0 flex items-center justify-center" style={{ backgroundColor: 'rgba(0, 0, 0, 0.1)' }}>
      <form onSubmit={handleSubmit} className="bg-gray-900 p-8 rounded-lg shadow-xl w-96">
        <h2 className="text-2xl text-white mb-6 text-center">Enter Password</h2>
        <div className="space-y-4">
          <input
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            className="w-full px-4 py-2 bg-gray-800 text-white rounded border border-gray-700 focus:border-blue-500 focus:outline-none"
            placeholder="Password"
            required
          />
          {error && (
            <p className="text-red-500 text-sm">{error}</p>
          )}
          <button
            type="submit"
            disabled={isLoading}
            className="w-full py-2 px-4 bg-blue-600 hover:bg-blue-700 text-white rounded transition duration-200 disabled:opacity-50"
          >
            {isLoading ? 'Authenticating...' : 'Enter'}
          </button>
        </div>
      </form>
    </div>
  );
};

export default PasswordAuth; 