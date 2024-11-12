#include <SFML/Graphics.hpp>
#include <cstdlib>
#include <ctime>

int main()
{
    int screenWidth = 800;
    int screenHeight = 450;
    sf::RenderWindow window(sf::VideoMode(screenWidth, screenHeight), "02_Array_Loop_Condition");

    const int numRect = 30;
    sf::RectangleShape rects[numRect];
    float speed[numRect];
    bool direction[numRect];

    srand(static_cast<unsigned int>(time(0)));

    for (int i = 0; i < numRect; i++)
    {
        float randomX = rand() % screenWidth;
        float randomY = rand() % screenHeight;

        float rectWidth = static_cast<float>(rand() % 30 + 10);
        float rectHeight = static_cast<float>(rand() % 30 + 10);

        unsigned int randomR = rand() % 255;
        unsigned int randomG = rand() % 255;
        unsigned int randomB = rand() % 255;
        sf::Color randomColor = sf::Color(randomR, randomG, randomB);

        rects[i] = sf::RectangleShape(sf::Vector2f{ rectWidth, rectHeight });
        rects[i].setFillColor(randomColor);
        rects[i].setPosition(sf::Vector2f{ randomX, randomY });

        speed[i] = static_cast<float>(rand() % 5 + 1) * 0.1f;
        direction[i] = true;
    }

    while (window.isOpen())
    {
        sf::Event event;
        while (window.pollEvent(event))
        {
            if (event.type == sf::Event::Closed)
                window.close();
        }

        for (int i = 0; i < numRect; i++)
        {
            sf::Vector2f pos = rects[i].getPosition();
            float rectWidth = rects[i].getSize().x;

            if (direction[i]) {
                pos.x += speed[i];
            } else {
                pos.x -= speed[i];
            }

            if (pos.x + rectWidth >= screenWidth) {
                direction[i] = false;
            } else if (pos.x <= 0) {
                direction[i] = true;
            }

            rects[i].setPosition(pos);
        }

        window.clear();

        for (int i = 0; i < numRect; i++)
        {
            window.draw(rects[i]);
        }

        window.display();
    }

    return 0;
}
