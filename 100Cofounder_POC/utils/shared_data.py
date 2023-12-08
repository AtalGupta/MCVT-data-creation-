import threading

# Global dictionary for track ID mapping
# This dictionary is used to map original track IDs to updated track IDs.
# Key: Original track ID (str)
# Value: Updated track ID (str)
track_id_mapping = {}

# Lock for thread-safe access to the dictionary
# This lock is used to ensure that the track_id_mapping dictionary is accessed in a thread-safe manner.
# It should be acquired before accessing or modifying the dictionary and released after the operation is complete.
track_id_mapping_lock = threading.Lock()