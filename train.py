from detecto import core
dataset = core.Dataset('/sample')
model = core.Model(['top_left', 'top_right', 'bottom_left', 'bottom_right'])
losses = model.fit(dataset, epochs=50, verbose=True, learning_rate=0.001)

model.save('id_card_4_corner.pth')