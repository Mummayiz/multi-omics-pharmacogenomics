"""Database module for Multi-Omics Platform"""

from .database import MultiOmicsDatabase

# Initialize database instance
db = MultiOmicsDatabase("multi_omics.db")

__all__ = ['db', 'MultiOmicsDatabase']
