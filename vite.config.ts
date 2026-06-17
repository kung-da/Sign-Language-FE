import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import { readFile } from "node:fs/promises";
import { extname, join } from "node:path";

export default defineConfig({
  plugins: [
    react(),
    {
      name: "mediapipe-wasm-dev-server",
      configureServer(server) {
        server.middlewares.use("/mediapipe-dev-wasm", async (request, response, next) => {
          const fileName = request.url
            ?.replace(/^\/+/, "")
            .replace(/^mediapipe-dev-wasm\//, "")
            .split("?")[0];
          if (!fileName) {
            next();
            return;
          }

          try {
            const filePath = join(process.cwd(), "node_modules", "@mediapipe", "tasks-vision", "wasm", fileName);
            const file = await readFile(filePath);
            response.setHeader(
              "Content-Type",
              extname(filePath) === ".wasm" ? "application/wasm" : "application/javascript",
            );
            response.end(file);
          } catch {
            next();
          }
        });
      },
    },
  ],
});
