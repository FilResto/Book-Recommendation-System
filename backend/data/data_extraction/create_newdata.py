import sys
import os

from extract_books import extract_ratings,extract_visualization,extract_userbook
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
from data.data_simulation.simulate_visualization import create_visual
from data.data_simulation.simualate_rating import create_ratings
from data.data_simulation.managing_genres_books import assign_random_genres

extract_userbook()
assign_random_genres()
create_visual()
extract_visualization()
create_ratings()
extract_ratings()