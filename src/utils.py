import gc
import psutil
import time
from pathlib import Path
from datetime import datetime
from config import Config

class MemoryManager:
    """Memory management utilities for Pi Zero 2 W"""
    
    def __init__(self, config: Config):
        self.config = config
        self.memory_threshold = config.memory_threshold
    
    def get_memory_usage(self):
        """Get current memory usage percentage"""
        try:
            return psutil.virtual_memory().percent / 100.0
        except Exception as e:
            print(f"Error getting memory usage: {e}")
            return 0.5  # Default to 50% if unable to determine
    
    def is_memory_available(self):
        """Check if memory usage is below threshold"""
        return self.get_memory_usage() < self.memory_threshold
    
    def force_cleanup(self):
        """Force garbage collection and cleanup"""
        try:
            gc.collect()
            return True
        except Exception as e:
            print(f"Error in force cleanup: {e}")
            return False
    
    def get_memory_info(self):
        """Get detailed memory information"""
        try:
            mem = psutil.virtual_memory()
            return {
                'total_mb': mem.total / (1024 * 1024),
                'available_mb': mem.available / (1024 * 1024),
                'used_mb': mem.used / (1024 * 1024),
                'percent': mem.percent,
                'free_mb': mem.free / (1024 * 1024)
            }
        except Exception as e:
            print(f"Error getting memory info: {e}")
            return None

class FileManager:
    """File management utilities for Pi Zero 2 W"""
    
    def __init__(self, config: Config):
        self.config = config
    
    def cleanup_old_images(self):
        """Delete oldest images if exceeding max limit"""
        try:
            image_files = sorted(
                list(self.config.image_dir.glob(f"{self.config.image_prefix}*.jpg")),
                key=lambda x: x.stat().st_mtime
            )
            
            if len(image_files) > self.config.max_images:
                files_to_delete = image_files[:(len(image_files) - self.config.max_images)]
                deleted_count = 0
                
                for file_path in files_to_delete:
                    try:
                        file_path.unlink()
                        deleted_count += 1
                    except Exception as e:
                        print(f"Error deleting {file_path}: {e}")
                
                print(f"Cleaned up {deleted_count} old images")
                return deleted_count
            
            return 0
            
        except Exception as e:
            print(f"Error in image cleanup: {e}")
            return 0
    
    def get_storage_info(self):
        """Get storage space information"""
        try:
            usage = psutil.disk_usage(self.config.data_dir)
            return {
                'total_mb': usage.total / (1024 * 1024),
                'used_mb': usage.used / (1024 * 1024),
                'free_mb': usage.free / (1024 * 1024),
                'percent': (usage.used / usage.total) * 100
            }
        except Exception as e:
            print(f"Error getting storage info: {e}")
            return None
    
    def ensure_directories(self):
        """Ensure all required directories exist"""
        try:
            self.config.data_dir.mkdir(exist_ok=True)
            self.config.image_dir.mkdir(exist_ok=True)
            self.config.logs_dir.mkdir(exist_ok=True)
            return True
        except Exception as e:
            print(f"Error creating directories: {e}")
            return False
    
    def get_image_count(self):
        """Get current number of stored images"""
        try:
            return len(list(self.config.image_dir.glob(f"{self.config.image_prefix}*.jpg")))
        except Exception as e:
            print(f"Error counting images: {e}")
            return 0

class SystemMonitor:
    """System monitoring utilities for Pi Zero 2 W"""
    
    def __init__(self, config: Config):
        self.config = config
        self.memory_manager = MemoryManager(config)
        self.file_manager = FileManager(config)
    
    def get_system_status(self):
        """Get comprehensive system status"""
        return {
            'timestamp': datetime.now().isoformat(),
            'memory': self.memory_manager.get_memory_info(),
            'storage': self.file_manager.get_storage_info(),
            'image_count': self.file_manager.get_image_count(),
            'memory_available': self.memory_manager.is_memory_available()
        }
    
    def should_skip_processing(self):
        """Determine if processing should be skipped due to resource constraints"""
        if not self.memory_manager.is_memory_available():
            print(f"Skipping processing: Memory usage above {self.config.memory_threshold*100}%")
            return True
        return False
    
    def log_system_status(self):
        """Log current system status"""
        status = self.get_system_status()
        if status['memory']:
            print(f"Memory: {status['memory']['percent']:.1f}% used "
                  f"({status['memory']['available_mb']:.0f}MB available)")
        if status['storage']:
            print(f"Storage: {status['storage']['percent']:.1f}% used "
                  f"({status['storage']['free_mb']:.0f}MB free)")
        print(f"Images stored: {status['image_count']}")

class PerformanceTimer:
    """Performance timing utility"""
    
    def __init__(self, operation_name="Operation"):
        self.operation_name = operation_name
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start timing"""
        self.start_time = time.time()
        return self
    
    def stop(self):
        """Stop timing and return duration"""
        if self.start_time is None:
            return 0.0
        
        self.end_time = time.time()
        duration = self.end_time - self.start_time
        return duration
    
    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        duration = self.stop()
        if duration > 1.0:  # Log slow operations
            print(f"{self.operation_name} took {duration:.2f}s")