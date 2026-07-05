/** @type {import('next').NextConfig} */
const nextConfig = {
  reactStrictMode: true,
  // 后端调用走 /api/chat route handler，由服务端转发，无需前端代理
};

export default nextConfig;
