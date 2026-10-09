// Page-level helpers: one readable error line when a load or the charts fail (FE-12), and a check that
// the chart library arrived (FE-16). The library comes from a CDN, so it can be missing.

export function showError(message) {
  let box = document.getElementById("page-error");
  if (!box) {
    box = document.createElement("div");
    box.id = "page-error";
    box.className = "page-error";
    box.setAttribute("role", "alert");
    const header = document.querySelector("header");
    if (header) header.insertAdjacentElement("afterend", box);
    else document.body.prepend(box);
  }
  box.textContent = message;
  box.hidden = false;
}

export function clearError() {
  const box = document.getElementById("page-error");
  if (box) box.hidden = true;
}

export function requireCharts() {
  if (typeof echarts === "undefined") {
    throw new Error("The charts did not load from cdn.jsdelivr.net. Check the connection and reload the page.");
  }
}
