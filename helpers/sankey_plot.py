import plotly.graph_objects as go

fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = ["1st. action", "2nd. action", "3rd. action", "4th. action", "5th. action", "Old approach (18 votes)", "New approach (108 votes)", "Both are good (19 votes)", "Both are bad (15 votes)"],
      color = ["blue", "blue", "blue", "blue", "blue", "orange", "orange", "orange", "orange", "orange"]
    ),
    link = dict(
      source = [0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4],
      target = [5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8, 5, 6, 7, 8] ,
      value = [13, 14, 3, 2, 1, 22, 0, 9, 3, 20, 7, 2, 0, 21, 10, 1, 1, 31, 0, 0],
      hovercolor=["midnightblue", "lightskyblue", "gold", "mediumturquoise", "lightgreen", "cyan"]
  ))])

fig.update_layout(title_text="Comparison with our older approach", font_size=20)
# fig.write_image("comparison_with_our_older_approach.pdf")
# fig.show()

fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = ["1st. action", "2nd. action", "3rd. action", "4th. action", "Very good (36 votes)", "Good (64 votes)", "Average (22 votes)", "Bad (6 votes)", "Very bad (0 votes)"],
      color = ["blue", "blue", "blue", "blue", "orange", "orange", "orange", "orange", "orange"]
    ),
    link = dict(
      source = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3],
      target = [4, 5, 6, 7, 8, 4, 5, 6, 7, 8, 4, 5, 6, 7, 8, 4, 5, 6, 7, 8, 5, 6, 7, 8] ,
      value = [16, 13, 3, 0, 0, 10, 18, 3, 1, 0, 3, 17, 12, 0, 0, 7, 16, 4, 5, 0],
      hovercolor=["midnightblue", "lightskyblue", "gold", "mediumturquoise", "lightgreen", "cyan"]
  ))])

fig.update_layout(title_text="NAO performance", font_size=20)
# fig.write_image("nao_performance.pdf")
# fig.show()


fig = go.Figure(data=[go.Sankey(
    node = dict(
      pad = 15,
      thickness = 20,
      line = dict(color = "black", width = 0.5),
      label = ["Action shape", "Action speed", "Very good (4 votes)", "Good (29 votes)", "Average (26 votes)", "Bad (5 votes)", "Very bad (0 votes)"],
      color = ["blue", "blue", "orange", "orange", "orange", "orange", "orange"]
    ),
    link = dict(
      source = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
      target = [2, 3, 4, 5, 6, 2, 3, 4, 5, 6] ,
      value = [3, 21, 8, 0, 0, 1, 8, 18, 5, 0],
      hovercolor=["midnightblue", "lightskyblue", "gold", "mediumturquoise", "lightgreen", "cyan"]
  ))])

fig.update_layout(title_text="QTrobot real-time performace", font_size=20)
fig.write_image("qt_live_performance.pdf")
fig.show()