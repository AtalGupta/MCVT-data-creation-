import threading

# Global dictionary for track ID mapping
track_id_mapping = {}

# Lock for thread-safe access to the dictionary
track_id_mapping_lock = threading.Lock()
