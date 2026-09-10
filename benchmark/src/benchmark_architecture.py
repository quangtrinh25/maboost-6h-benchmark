"""Generate the architecture diagram locally; no figure asset is distributed."""
def draw_architecture(path):
    import matplotlib.pyplot as plt
    from matplotlib.patches import FancyBboxPatch
    from v13_guard import assert_safe_output_path
    path = assert_safe_output_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({'pdf.fonttype':42,'font.size':10}):
        fig, ax = plt.subplots(figsize=(12,5))
        ax.set(xlim=(0,12),ylim=(0,5));ax.axis('off')
        nodes = [(0.2,2,'0–6 h measurements\n40 dynamic variables'),(3.1,3.6,'Mamba encoder\n512-dimensional state'),
                 (3.1,1.9,'200 temporal summaries\n12 static covariates'),(6.2,3.6,'Hybrid pathway'),
                 (6.2,1.9,'XGBoost-flat'),(6.2,0.3,'Flat-head'),(9.3,2,'Source-selected fusion\nCalibration / threshold')]
        for x,y,label in nodes:
            ax.add_patch(FancyBboxPatch((x,y),2.35,.8,boxstyle='round,pad=0.05',facecolor='#edf4fa',edgecolor='#36648b'))
            ax.text(x+1.175,y+.4,label,ha='center',va='center')
        for a,b in [(0,1),(0,2),(1,3),(2,3),(2,4),(2,5),(3,6),(4,6),(5,6)]:
            x,y,_=nodes[a];u,v,_=nodes[b]
            ax.annotate('',xy=(u,v+.4),xytext=(x+2.35,y+.4),arrowprops={'arrowstyle':'->','color':'#536878'})
        ax.text(6,.02,'Fit preprocessing, fusion, calibration and thresholds on source data only.',ha='center')
        fig.savefig(path,bbox_inches='tight');plt.close(fig)
    return path
