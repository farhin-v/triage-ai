import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.database.connections import create_tables

print("Testing database connection...")
create_tables()