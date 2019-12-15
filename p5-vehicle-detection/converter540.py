import moviepy.editor as mp
clip = mp.VideoFileClip("output_v5_rs.mp4")
clip_resized = clip.resize(height=540) # make the height 360px ( According to moviePy documenation The width is then computed so that the width/height ratio is conserved.)
clip_resized.write_videofile("movie_resized.mp4")
