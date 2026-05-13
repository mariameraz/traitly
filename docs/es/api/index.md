---
 hide:
   - navigation
   - toc
---

<div class="animate" markdown>

# Referencias API ⊹ ࣪ ˖

Aquí encontrarás la referencia completa de todos los módulos, clases y funciones disponibles en el paquete Traitly.

---

Traitly está organizado en los siguientes módulos:

- **fruit_phenotyping/** — lógica principal de análisis: morfología, color y segmentación de imágenes de frutos.
- **shiny_app/** — aplicación web interactiva desarrollada con Shiny.
- **pdf/** — funciones para extraer imágenes de archivos PDF.
- **utils/** — funciones auxiliares y constantes compartidas entre módulos.
- **package_data/** — modelos preentrenados para la detección de etiquetas y referencias de tamaño.
- **cli.py** — punto de entrada para ejecutar Traitly desde la línea de comandos.

??? tree-diagram "Estructura del Proyecto"
    ```
    src/traitly/
    │
    ├── fruit_phenotyping/
    │   │
    │   ├── __init__.py
    │   ├── analysis_parameters.py
    │   ├── color_analysis.py
    │   ├── color_plot.py
    │   ├── external_analysis.py
    │   ├── fruit_config.py
    │   ├── geometry.py
    │   ├── internal_analysis.py
    │   ├── mask.py
    │   ├── processing.py
    │   ├── results_image.py
    │   └── symmetry.py
    │
    ├── package_data/
    │   │
    │   └── models/
    │       ├── label.pt
    │       └── size_reference.pt
    │
    ├── pdf/
    │   ├── __init__.py
    │   └── convert_pdf.py
    │
    ├── shiny_app/
    │   │
    │   ├── www/
    │   │   └── parameters.json
    │   └── app.py
    │
    ├── utils/
    │   │
    │   ├── __init__.py
    │   ├── basic_functions.py
    │   ├── calibration.py
    │   ├── constants.py
    │   └── label.py
    │
    ├── __init__.py
    └── cli.py
    
    ```
</div>

---

Lista de funciones y clases en el paquete Traitly:

!!! warning
    La documentación de módulos, clases y funciones está disponible únicamente en inglés.

<link rel="stylesheet" href="https://cdn.datatables.net/1.13.6/css/jquery.dataTables.min.css">
<script src="https://code.jquery.com/jquery-3.7.0.min.js"></script>
<script src="https://cdn.datatables.net/1.13.6/js/jquery.dataTables.min.js"></script>

<style>
  table#api-table { width: 100% !important; font-size: 0.85rem; }
  table#api-table thead th { background-color: var(--md-primary-fg-color); color: white; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.75rem; font-weight: 600; }
  .badge-function { background-color: #e3f2fd; color: #1565c0; }
  .badge-class { background-color: #fce4ec; color: #880e4f; }
  #api-filters { display: flex; align-items: center; gap: 12px; margin-bottom: 1rem; flex-wrap: wrap; }
  #api-filters label { font-size: 0.85rem; }
  #api-filters select { font-size: 0.85rem; padding: 4px 8px; border: 1px solid #ccc; border-radius: 4px; }
</style>

<div id="api-filters">
  <label for="module-filter">Módulo:</label>
  <select id="module-filter">
    <option value="">All</option>
  </select>

  <label for="type-filter">Tipo:</label>
  <select id="type-filter">
    <option value="">All</option>
    <option value="Function">Function</option>
    <option value="Class">Class</option>
  </select>
</div>

<table id="api-table" class="display" style="width:100%">
  <thead>
    <tr><th>Tipo</th><th>Nombre</th><th>Módulo</th><th>Descripción</th></tr>
  </thead>
  <tbody id="api-tbody"></tbody>
</table>

<script>
const modulePageMap = {
  "traitly.fruit_phenotyping.mask": "../../docstrings/modules/mask",
  "traitly.fruit_phenotyping.geometry": "../../docstrings/modules/geometry",
  "traitly.fruit_phenotyping.fruit_config": "../../docstrings/modules/fruit_config",
  "traitly.fruit_phenotyping.processing": "../../docstrings/modules/processing",
  "traitly.fruit_phenotyping.color_analysis": "../../docstrings/modules/color_analysis",
  "traitly.fruit_phenotyping.symmetry": "../../docstrings/modules/symmetry",
  "traitly.fruit_phenotyping.color_plot": "../../docstrings/modules/color_plot",
  "traitly.fruit_phenotyping.analysis_parameters": "../../docstrings/modules/analysis_parameters",
  "traitly.utils.basic_functions": "../../docstrings/modules/basic_functions",
  "traitly.utils.calibration": "../../docstrings/modules/calibration",
  "traitly.utils.label": "../../docstrings/modules/label",
  "traitly.pdf.convert_pdf": "../../docstrings/modules/convert_pdf",
  "traitly.cli": "../../docstrings/modules/cli",
};

fetch("../../docstrings/api_data.json")
  .then(r => r.json())
  .then(function(data) {
    const tbody = document.getElementById("api-tbody");

    const modules = [...new Set(data.map(r => r.module.split(".").slice(0, 2).join(".")))].sort();
    const moduleSelect = document.getElementById("module-filter");
    modules.forEach(function(m) {
      moduleSelect.innerHTML += `<option value="${m}">${m}</option>`;
    });

    // Render rows
    data.forEach(function(row) {
      const badgeClass = row.type === "Class" ? "badge-class" : "badge-function";
      const page = modulePageMap[row.module];
      const nameCell = page
        ? `<a href="${page}/#${row.module}.${row.name.toLowerCase()}"><code>${row.name}</code></a>`
        : `<code>${row.name}</code>`;

      tbody.innerHTML += `<tr>
        <td><span class="badge ${badgeClass}">${row.type}</span></td>
        <td>${nameCell}</td>
        <td><code>${row.module}</code></td>
        <td>${row.description}</td>
      </tr>`;
    });

    const table = $("#api-table").DataTable({
      pageLength: 20,
      lengthMenu: [10, 20, 50, 100],
      order: [[1, "asc"]],
      search: { smart: false }
    });
    
    $.fn.dataTable.ext.search.push(function(settings, data, dataIndex) {
      const moduleVal = document.getElementById("module-filter").value;
      const typeVal = document.getElementById("type-filter").value;
      const moduleMatch = !moduleVal || data[2].includes(moduleVal);
      const typeMatch = !typeVal || data[0].includes(typeVal);
      return moduleMatch && typeMatch;
    });
    
    document.getElementById("module-filter").addEventListener("change", function() {
      table.draw();
    });
    
    document.getElementById("type-filter").addEventListener("change", function() {
      table.draw();
    });
  });
</script>
