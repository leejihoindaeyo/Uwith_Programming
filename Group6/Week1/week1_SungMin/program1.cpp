#include <SFML/Graphics.hpp>
#include <iostream>
int main()
{
	sf::RenderWindow window(sf::VideoMode(800, 450), "SFML works!");

	float rectWidth = 20.0f;
	float rectHeight = 10.0f;
	float rectPositionX = 10.0;
	float rectPositionY = 50.0;
	bool flagRec = true;

	sf::RectangleShape shape(sf::Vector2f{ rectWidth , rectHeight });
	shape.setFillColor(sf::Color::Red);

	sf::CircleShape shape1(50.0f);
	shape1.setFillColor(sf::Color::Cyan);
	float CirRad = 50.0f;
	float CirPositionX = 10.0;
	float CirPositionY = 150.0;
	bool flagCir = true;

	float windowwidth = window.getSize().x;
	float windowHeight = window.getSize().y;

	while (window.isOpen())
	{
		sf::Event event;
		while (window.pollEvent(event))
		{
			if (event.type == sf::Event::Closed)
				window.close();
		}
		
		if (rectPositionX >= (windowwidth - rectWidth) || rectPositionX <= 0) {
			flagRec = !flagRec;
		}

		if (flagRec) {
			rectPositionX += 0.05f;
		}
		else {
			rectPositionX -= 0.05f;
		}

		if (CirPositionX >= (windowwidth - CirRad) || CirPositionX <= 0) {
			flagCir = !flagCir;
		}

		if (flagCir) {
			CirPositionX += 0.1f;
		}
		else {
			CirPositionX -= 0.1f;
		}

		shape.setPosition(rectPositionX, rectPositionY);
		shape1.setPosition(CirPositionX, CirPositionY);


		window.clear();
		window.draw(shape);
		window.draw(shape1);
		window.display();
	}
}