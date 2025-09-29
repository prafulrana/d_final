FROM node:18-alpine

# Create app directory
WORKDIR /app

# Copy package files
COPY package*.json ./

# Install dependencies
RUN npm install

# Copy app source
COPY . .

# Create tmp directory for video files
RUN mkdir -p /tmp

# Expose port
EXPOSE 3001

# Start the application
CMD ["npm", "start"]