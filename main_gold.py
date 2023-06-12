import gold.read as gr
from config import raw as config
import matplotlib

adj_close = gr.read_csv(config['adj_close']['path'])
((adj_close/adj_close.iloc[0])*100).plot(figsize = (10,10))



