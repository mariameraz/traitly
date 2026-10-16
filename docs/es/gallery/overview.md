---
hide:
  - navigation
  - toc

---

<style>

.md-typeset .grid.cards > ul {
    display: grid !important;
    grid-template-columns: repeat(2, minmax(0, 1fr)) !important; /* two columns */
    gap: 1rem;
    align-items: stretch;
}

/* flexible cards */
.md-typeset .grid.cards > ul > li {
    display: flex;
    flex-direction: column;
    margin: 0 !important;
    box-sizing: border-box;
}

/* Image ratio 16:9 */
.md-typeset .grid.cards img {
    width: 100%;
    aspect-ratio: 16 / 9;
    object-fit: contain;
    display: block;
}

.md-typeset .grid.cards > ul > li > p:last-child {
    margin-top: auto;
}

.md-typeset .grid.cards .card-title-big {
    display: block;
    text-align: center;
    font-weight: 600;
}

/* Reduce cols to 1 in small windows */
@media (max-width: 800px) {
    .md-typeset .grid.cards > ul {
        grid-template-columns: 1fr !important;
    }
}
</style>


<div class="animate" markdown>

# Galería ⊹ ࣪ ˖

Aquí encontraras ejemplos de imagenes analizadas con Traitly. Haz click en el título para ver el código detrás de cada imagen.

---

<br>

</div>

<div class="grid cards" markdown>

-   ![Beans](https://i.swiftimg.com/variants/r4JRIOum/display.webp)

    ---

    [Análisis morfológico de frijoles](../scripts/bean.md){ .card-title-big }

-   ![Gehler dataset](https://i.swiftimg.com/variants/GpMXfYll/display.webp)

    ---

    [Corrección de color de una imagen del dataset Gehler](../scripts/color_correction.md){ .card-title-big }


-   ![Strawberry color analysis](https://i.swiftimg.com/variants/q1pzqApz/display.webp)

    ---

    [Análisis de color de fresas](../scripts/strawberry.md){ .card-title-big }


-   ![Potato](https://i.swiftimg.com/originals/1Yo0vcOW.png)

    ---

    [Análisis morfológico de papas](../scripts/potato.md){ .card-title-big }

-   ![Cucumber](https://i.swiftimg.com/variants/lofO2pit/display.webp)

    ---

    [Análisis de rodajas de pepinos](../scripts/cucumber.md){ .card-title-big }


-   ![Gehler dataset 2](https://i.swiftimg.com/variants/H3V1qXRj/display.webp)

    ---

    [Corrección de color de una imagen del dataset Gehler](../scripts/color_correction_photo2.md){ .card-title-big }

</div>
