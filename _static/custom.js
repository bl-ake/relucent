/* Remove the intermediate py:class entries from the sidebar navigation,
   promoting their method children (toctree-l3) directly to toctree-l2 level
   so methods appear directly under the page title. */
document.addEventListener('DOMContentLoaded', function () {
    var nav = document.querySelector('.wy-menu-vertical');
    if (!nav) return;

    nav.querySelectorAll('.toctree-l2').forEach(function (l2) {
        var inner = l2.querySelector(':scope > ul');
        if (!inner) return;

        // Collect promoted children in a fragment to preserve insertion order.
        var frag = document.createDocumentFragment();
        inner.querySelectorAll(':scope > .toctree-l3').forEach(function (l3) {
            l3.classList.replace('toctree-l3', 'toctree-l2');
            frag.appendChild(l3);
        });
        l2.parentNode.insertBefore(frag, l2.nextSibling);
        l2.remove();
    });
});
