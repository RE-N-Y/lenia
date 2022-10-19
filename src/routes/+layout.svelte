<svelte:head>
	<style>
		@import url("https://fonts.googleapis.com/css?family=Quicksand&display=swap");
	</style>
</svelte:head>

<script>
	import '../app.css';
	import 'material-icons/iconfont/material-icons.css'
</script>

<div class="app">
	<main class="h-screen">
		<slot />
	</main>
</div>
